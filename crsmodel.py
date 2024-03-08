import torch, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from sklearn.metrics import mutual_info_score

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x):
        assert self.dim == x.shape[1]
        e = torch.matmul(torch.tanh(torch.matmul(x, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        return torch.matmul(attention, x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(batch_size * n_heads, seq_len, dim_per_head)
            return tensor

        if key is None and value is None:
            key = value = query
        elif value is None:
            value = key

        _, key_len, dim = key.size()
        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        mask = (mask.view(batch_size, 1, -1, key_len).repeat(1, n_heads, 1, 1).expand(batch_size, n_heads, query_len, key_len).view(batch_size * n_heads, query_len, key_len))
        assert mask.shape == dot_prod.shape
        dot_prod *= mask
        dot_prod.masked_fill_((mask == 0), -NEAR_INF_FP16 if dot_prod.dtype is torch.float16 else -NEAR_INF)
        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)
        attentioned = attn_weights.bmm(v)
        attentioned = (attentioned.type_as(query).view(batch_size, n_heads, query_len, dim_per_head).transpose(1, 2).contiguous().view(batch_size, query_len, dim))
        out = self.out_lin(attentioned)
        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0.0):
        super().__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)
        x = self.lin2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, embedding_size, ffn_size, attention_dropout=0.0, relu_dropout=0.0, dropout=0.0):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, encoder_mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=encoder_mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        return tensor


class TransformerDecoder4KGLayer(nn.Module):
    def __init__(self, n_heads, embedding_size, ffn_size, attention_dropout=0.0, relu_dropout=0.0, dropout=0.0):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)
        self.self_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.encoder_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.encoder_db_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2_db = nn.LayerNorm(embedding_size)
        self.encoder_kg_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2_kg = nn.LayerNorm(embedding_size)
        self.encoder_user_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2_user = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, decoder_mask, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output, db_encoder_mask, user_encoder_output, user_encoder_mask):
        residual = x
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)
        x = x + residual
        x = _normalize(x, self.norm1)
        # ori0: start
        residual = x
        x = self.encoder_db_attention(query=x, key=db_encoder_output, value=db_encoder_output, mask=db_encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_db)
        residual = x
        x = self.encoder_kg_attention(query=x, key=kg_encoder_output, value=kg_encoder_output, mask=kg_encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_kg)
        residual = x
        x = self.encoder_user_attention(query=x, key=user_encoder_output, value=user_encoder_output, mask=user_encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_user)
        # ori0: end
        residual = x
        x = self.encoder_attention(query=x, key=encoder_output, value=encoder_output, mask=encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm3)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.n_heads = args.n_heads
        self.embedding_size = args.embedding_size
        self.ffn_size = args.ffn_size
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.relu_dropout = args.relu_dropout
        self.max_c_length = args.max_c_length
        self.special_wordIdx = args.special_wordIdx
        self.n_mood = args.n_mood
        self.device = args.device
        self.drop = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.mood_attn = SelfAttentionLayer(self.embedding_size, self.embedding_size)
        self.mood_attn_fc = nn.Linear(self.embedding_size, self.n_mood)
        self.mood_embeddings = nn.Embedding(self.n_mood, self.embedding_size)
        nn.init.normal_(self.mood_embeddings.weight, mean=0, std=self.embedding_size ** -0.5)
        self.position_embeddings = nn.Embedding(self.max_c_length, self.embedding_size)
        nn.init.normal_(self.position_embeddings.weight, 0, self.embedding_size ** -0.5)
        self.layers = nn.ModuleList([TransformerEncoderLayer(self.n_heads, self.embedding_size, self.ffn_size, attention_dropout=self.attention_dropout, relu_dropout=self.relu_dropout, dropout=self.dropout) for _ in range(self.n_layers)])

    def forward(self, sum_embeddings, context_vector, context_mask, context_pos, context_vm):
        context_emb = sum_embeddings[context_vector]
        last_row = -1
        last_col = -1
        for indice in torch.nonzero(context_mask == self.special_wordIdx['<mood>']):
            if indice[0] == last_row:
                context_emb[[indice[0], indice[1]]] = self.mood_embeddings(torch.argmax(self.mood_attn_fc(self.mood_attn(context_emb[indice[0], last_col + 1:indice[1]]))))
                last_col = indice[1]
            else:
                context_emb[[indice[0], indice[1]]] = self.mood_embeddings(torch.argmax(self.mood_attn_fc(self.mood_attn(context_emb[indice[0], 0:indice[1]]))))
                last_row = indice[0]
                last_col = indice[1]
        context_emb *= torch.tensor(np.sqrt(self.embedding_size)).to(self.device)
        context_emb += self.position_embeddings(context_pos)
        context_emb += sum_embeddings[context_mask]
        context_emb = self.drop(self.layer_norm(context_emb))
        for i in range(self.n_layers):
            context_emb = self.layers[i](context_emb, context_vm)
        return context_emb


class TransformerDecoder4KG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.embedding_size = args.embedding_size
        self.ffn_size = args.ffn_size
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.relu_dropout = args.relu_dropout
        self.max_c_length = args.max_c_length
        self.special_wordIdx = args.special_wordIdx
        self.batch_size = args.batch_size
        self.device = args.device
        self.drop = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.position_embeddings = nn.Embedding(self.max_c_length, self.embedding_size)
        nn.init.normal_(self.position_embeddings.weight, 0, self.embedding_size ** -0.5)
        self.layers = nn.ModuleList([TransformerDecoder4KGLayer(self.n_heads, self.embedding_size, self.ffn_size, attention_dropout=self.attention_dropout, relu_dropout=self.relu_dropout, dropout=self.dropout) for _ in range(self.n_layers)])

    def forward(self, sum_embeddings, predict_vector, encoder_latent_emb, con_graph_fc_emb, db_graph_fc_emb, user_graph_fc_emb, context_mask):
        encoder_latent_emb_mask = (context_mask != self.special_wordIdx['<pad>'])
        con_graph_fc_emb_mask = (context_mask != self.special_wordIdx['<concept>'])
        db_graph_fc_emb_mask = (context_mask != self.special_wordIdx['<dbpedia>'])
        user_graph_fc_emb_mask = (context_mask != self.special_wordIdx['<user>'])
        predict_pos = torch.arange(predict_vector.shape[1], dtype=torch.long).unsqueeze(0).expand(self.batch_size, predict_vector.shape[1]).to(self.device)
        predict_emb = sum_embeddings[predict_vector]
        predict_emb *= torch.tensor(np.sqrt(self.embedding_size)).to(self.device)
        predict_emb += self.position_embeddings(predict_pos)
        predict_emb = self.drop(self.layer_norm(predict_emb))
        predict_vm = torch.tril(torch.ones(predict_emb.size(0), predict_emb.size(1), predict_emb.size(1))).to(self.device)
        for layer in self.layers:
            predict_emb = layer(predict_emb, predict_vm, encoder_latent_emb, encoder_latent_emb_mask, con_graph_fc_emb, con_graph_fc_emb_mask, db_graph_fc_emb, db_graph_fc_emb_mask, user_graph_fc_emb, user_graph_fc_emb_mask)
        return predict_emb


class Bert4KGModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.batch_size = args.batch_size
        self.n_user = args.n_user
        self.n_concept = args.n_concept
        self.n_dbpedia = args.n_dbpedia
        self.n_relations = args.n_relations
        self.batch_size = args.batch_size
        self.n_bases = args.n_bases
        self.word2wordEmb = args.word2wordEmb
        self.special_wordIdx = args.special_wordIdx
        self.hidden_dim = args.hidden_dim
        self.embedding_size = args.embedding_size
        self.max_c_length = args.max_c_length
        self.max_r_length = args.max_r_length
        self.device = args.device
        # 生成部分db_attn，encoder_states_kg部分的参数
        self.concept_GCN = GCNConv(self.hidden_dim, self.hidden_dim)
        self.dbpedia_edge_idx = args.dbpedia_edge_list[:, :2].t()
        self.dbpedia_edge_type = args.dbpedia_edge_list[:, 2]
        self.concept_edge_sets = args.concept_edge_sets
        self.dbpedia_RGCN = RGCNConv(self.n_dbpedia + self.n_user, self.hidden_dim, self.n_relations, num_bases=self.n_bases)
        self.concept_embeddings = nn.Embedding(self.n_concept, self.hidden_dim)
        nn.init.normal_(self.concept_embeddings.weight, mean=0, std=self.embedding_size ** -0.5)
        nn.init.constant_(self.concept_embeddings.weight[self.special_wordIdx['<pad>']], 0)
        self.con_graph_attn = SelfAttentionLayer(self.hidden_dim, self.hidden_dim)
        self.db_graph_attn = SelfAttentionLayer(self.hidden_dim, self.hidden_dim)
        self.user_graph_attn = SelfAttentionLayer(self.hidden_dim, self.hidden_dim)
        # info_loss部分的参数
        self.user_con_info_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.user_db_info_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.db_con_info_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.con_output = nn.Linear(self.hidden_dim, self.n_concept)
        self.db_output = nn.Linear(self.hidden_dim, self.n_dbpedia)
        self.user_output = nn.Linear(self.hidden_dim, self.n_user)
        self.mse_loss = nn.MSELoss(size_average=False, reduce=False)
        # mutual_loss部分的参数
        self.mine_layers = nn.Sequential(nn.Linear(3 * self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
        # rec_loss部分的参数
        self.user_db_con_fc = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.graph_rec_output = nn.Linear(self.hidden_dim, self.n_dbpedia)
        self.criterion_loss = nn.CrossEntropyLoss(reduce=False)
        # rec2_loss部分的参数
        self.graph_latent_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.encoder = TransformerEncoder(args)
        self.encoder_latent_fc = nn.Linear(self.embedding_size, self.hidden_dim)
        self.encoder_graph_latent_fc = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.encoder_graph_rec_output = nn.Linear(self.hidden_dim, self.n_dbpedia)
        # sum_embeddings
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size, self.special_wordIdx['<pad>'])
        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.word2wordEmb))
        self.dbpedia_embeddings_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.concept_embeddings_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.relation_embeddings = nn.Embedding(self.n_relations, self.embedding_size)
        nn.init.normal_(self.relation_embeddings.weight, mean=0, std=self.embedding_size ** -0.5)
        nn.init.constant_(self.relation_embeddings.weight[self.special_wordIdx['<pad>']], 0)
        self.user_embeddings_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        # gen_loss部分的参数
        self.con_graph_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.db_graph_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.user_graph_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.con_graph_attn_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.db_graph_attn_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.user_graph_attn_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.decoder = TransformerDecoder4KG(args)
        self.decoder_graph_latent_fc = nn.Linear(4 * self.embedding_size, self.embedding_size)
        # for name, param in self.named_parameters():
        #     print(f"Module: {name}, Parameters: {param.numel()}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters in the model: {total_params}")

    def forward(self, userIdx, dbpediaId, context_vector, context_mask, context_pos, context_vm, response_vector, concept_vector, dbpedia_vector, user_vector):
        # 获得基础特征
        con_nodes_features = self.concept_GCN(self.concept_embeddings.weight, self.concept_edge_sets)
        dbpedia_nodes_features = self.dbpedia_RGCN(None, self.dbpedia_edge_idx, self.dbpedia_edge_type)
        db_nodes_features = dbpedia_nodes_features[:-self.n_user]
        user_nodes_features = dbpedia_nodes_features[-self.n_user:]
        # 也是基础特征
        db_graph_attn_emb = torch.stack([self.db_graph_attn(db_nodes_features[[b[0] for b in row[row.nonzero()].tolist()]]) for row in torch.where(context_mask == self.special_wordIdx['<dbpedia>'], context_vector - self.vocab_size - self.n_concept, torch.tensor(0))])
        con_graph_attn_emb = torch.stack([self.con_graph_attn(con_nodes_features[[b[0] for b in row[row.nonzero()].tolist()]]) for row in torch.where(context_mask == self.special_wordIdx['<concept>'], context_vector - self.vocab_size, torch.tensor(0))])
        user_graph_attn_emb = torch.stack([self.user_graph_attn(user_nodes_features[[b[0] for b in row[row.nonzero()].tolist()]]) for row in torch.where(context_mask == self.special_wordIdx['<user>'], context_vector - self.vocab_size - self.n_concept - self.n_dbpedia - self.n_relations, torch.tensor(0))])
        # info_loss
        user_db_info_emb = self.user_db_info_fc(torch.cat([user_graph_attn_emb, db_graph_attn_emb], dim=-1))
        user_con_info_emb = self.user_con_info_fc(torch.cat([user_graph_attn_emb, con_graph_attn_emb], dim=-1))
        db_con_info_emb = self.db_con_info_fc(torch.cat([db_graph_attn_emb, con_graph_attn_emb], dim=-1))
        con_scores = F.linear(user_db_info_emb, con_nodes_features, self.con_output.bias)
        db_scores = F.linear(user_con_info_emb, db_nodes_features, self.db_output.bias)
        user_scores = F.linear(db_con_info_emb, user_nodes_features, self.user_output.bias)
        info_db_loss = torch.mean(torch.sum(self.mse_loss(db_scores, dbpedia_vector.float()), dim=-1))
        info_con_loss = torch.mean(torch.sum(self.mse_loss(con_scores, concept_vector.float()), dim=-1))
        info_user_loss = torch.mean(torch.sum(self.mse_loss(user_scores, user_vector.float()), dim=-1))
        # mutual_loss
        tiled_x = torch.cat([user_graph_attn_emb, user_graph_attn_emb], dim=0)
        concat_y = torch.cat([db_graph_attn_emb, db_graph_attn_emb[torch.randperm(self.batch_size)]], dim=0)
        concat_z = torch.cat([con_graph_attn_emb, con_graph_attn_emb[torch.randperm(self.batch_size)]], dim=0)
        inputs = torch.cat([tiled_x, concat_y, concat_z], dim=1)
        logits = self.mine_layers(inputs)
        pred_xyz = logits[:self.batch_size]
        pred_x_y_z = logits[self.batch_size:]
        mutual_loss = - np.log2(np.exp(1)) * (torch.mean(pred_xyz) - torch.log(torch.mean(torch.exp(pred_x_y_z))))
        info_loss = mutual_loss + info_db_loss
        # 通过user_emb和db_nodes_features计算rec_scores，对比labels得到rec_loss
        graph_latent_emb = self.user_db_con_fc(torch.cat([user_graph_attn_emb, con_graph_attn_emb, db_graph_attn_emb], dim=-1))
        graph_rec_scores = F.linear(graph_latent_emb, db_nodes_features, self.graph_rec_output.bias)
        graph_rec_loss = torch.mean(torch.sum(self.criterion_loss(graph_rec_scores, dbpediaId) * (dbpediaId != 0).to(self.device)))

        # 计算gen_scores和preds--------可以把历史记录的movie_fc加上--------------------|##|Aab******#|Bc******#|Ac******#|Bc********#|-------------
        # sum_embeddings
        dbpedia_embeddings = self.dbpedia_embeddings_fc(db_nodes_features)
        user_embeddings = self.user_embeddings_fc(user_nodes_features)
        concept_embeddings = self.concept_embeddings_fc(con_nodes_features)
        sum_embeddings = torch.cat([self.word_embeddings.weight, concept_embeddings, dbpedia_embeddings, self.relation_embeddings.weight, user_embeddings], dim=0)
        # encoder
        encoder_latent_emb = self.encoder(sum_embeddings, context_vector, context_mask, context_pos, context_vm)
        divisor = (context_mask != self.special_wordIdx['<pad>']).type_as(encoder_latent_emb).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
        encoder_latent_emb_ = encoder_latent_emb.sum(dim=1) / divisor
        encoder_graph_latent_emb = self.encoder_graph_latent_fc(torch.cat([self.graph_latent_fc(graph_latent_emb), self.encoder_latent_fc(encoder_latent_emb_)], dim=-1))
        encoder_graph_rec_scores = F.linear(encoder_graph_latent_emb, db_nodes_features, self.encoder_graph_rec_output.bias)
        encoder_graph_rec_loss = torch.sum(self.criterion_loss(encoder_graph_rec_scores, dbpediaId) * (dbpediaId != 0).to(self.device))

        # decoder
        db_graph_fc_emb = self.db_graph_fc(db_nodes_features[torch.where(context_mask == self.special_wordIdx['<dbpedia>'], context_vector - self.vocab_size - self.n_concept, torch.tensor(0))])
        con_graph_fc_emb = self.con_graph_fc(con_nodes_features[torch.where(context_mask == self.special_wordIdx['<concept>'], context_vector - self.vocab_size, torch.tensor(0))])
        user_graph_fc_emb = self.user_graph_fc(user_nodes_features[torch.where(context_mask == self.special_wordIdx['<user>'], context_vector - self.vocab_size - self.n_concept - self.n_dbpedia - self.n_relations, torch.tensor(0))])
        db_graph_attn_fc_emb = self.db_graph_attn_fc(db_graph_attn_emb)
        con_graph_attn_fc_emb = self.con_graph_attn_fc(con_graph_attn_emb)
        user_graph_attn_fc_emb = self.user_graph_attn_fc(con_graph_attn_emb)
        if response_vector is None:
            predict_vector = (userIdx + self.vocab_size + self.n_concept + self.n_dbpedia + self.n_relations).view(self.batch_size, 1)
            for idx in range(self.max_r_length):
                decoder_latent_emb = self.decoder(sum_embeddings, predict_vector, encoder_latent_emb, con_graph_fc_emb, db_graph_fc_emb, user_graph_fc_emb, context_mask)
                last_token_emb = decoder_latent_emb[:, -1:, :]
                decoder_graph_latent_fc_emb = self.decoder_graph_latent_fc(torch.cat([con_graph_attn_fc_emb.unsqueeze(1), db_graph_attn_fc_emb.unsqueeze(1), user_graph_attn_fc_emb.unsqueeze(1), last_token_emb], dim=-1))
                gen_scores = F.linear(decoder_graph_latent_fc_emb, sum_embeddings[:self.vocab_size + self.n_concept])
                _, last_token = gen_scores.max(dim=-1)
                predict_vector = torch.cat([predict_vector, last_token], dim=1)
                if ((predict_vector == self.special_wordIdx['<eos>']).sum(dim=1) > 0).sum().item() == self.batch_size:
                    break
            predict_vector = predict_vector[:, 1:]
            gen_loss = None
        else:
            decoder_latent_emb = self.decoder(sum_embeddings, response_vector, encoder_latent_emb, con_graph_fc_emb, db_graph_fc_emb, user_graph_fc_emb, context_mask)
            decoder_graph_latent_fc_emb = self.decoder_graph_latent_fc(torch.cat([con_graph_attn_fc_emb.unsqueeze(1).repeat(1, self.max_r_length, 1), db_graph_attn_fc_emb.unsqueeze(1).repeat(1, self.max_r_length, 1), user_graph_attn_fc_emb.unsqueeze(1).repeat(1, self.max_r_length, 1), decoder_latent_emb], dim=-1))
            gen_scores = F.linear(decoder_graph_latent_fc_emb, sum_embeddings[:self.vocab_size + self.n_concept])
            gen_loss = torch.mean(self.criterion_loss(gen_scores[:, :-1].reshape(-1, self.vocab_size + self.n_concept), response_vector[:, 1:].reshape(-1)))
            predict_vector = None
        return info_loss, graph_rec_scores, graph_rec_loss, encoder_graph_rec_scores, encoder_graph_rec_loss, predict_vector, gen_loss

    def freeze_kg(self, freezeKG):
        if freezeKG:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(), self.concept_embeddings.parameters(), self.con_graph_attn.parameters(), self.db_graph_attn.parameters(), self.db_graph_attn.parameters(), self.user_graph_attn.parameters(), self.user_db_con_fc.parameters(), self.graph_rec_output.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False
            print(f"Freeze parameters in the model")
        else:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(), self.concept_embeddings.parameters(), self.con_graph_attn.parameters(), self.db_graph_attn.parameters(), self.db_graph_attn.parameters(), self.user_graph_attn.parameters(), self.user_db_con_fc.parameters(), self.graph_rec_output.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = True
            print(f"UnFreeze parameters in the model")

    def save_model(self, tag):
        if tag == "rec":
            torch.save(self.state_dict(), 'rec_net_parameter.pkl')
        else:
            torch.save(self.state_dict(), 'gen_net_parameter.pkl')

    def load_model(self, tag):
        if tag == "rec":
            self.load_state_dict(torch.load('rec_net_parameter.pkl'), strict=False)
        else:
            self.load_state_dict(torch.load('gen_net_parameter.pkl'), strict=False)

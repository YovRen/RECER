from collections import defaultdict
import torch
import json
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from transformers.models.gpt2.modeling_gpt2 import GPT2Model,GPT2Block


class GPT24KG(GPT2Model):
    def __init__(self, config):
        config.add_cross_attention=True
        super().__init__(config)
        config.add_cross_attention=False
        self.config = config
        self.embd_pdrop = config.embd_pdrop
        self.vocab_size = config.vocab_size
        self.n_user = 1075
        self.n_concept = 29308
        self.n_dbpedia = 64363
        self.n_relations = 46
        self.n_bases = 8
        self.dim = 120
        # 生成部分db_attn，encoder_states_kg部分的参数
        self.dbpedia_RGCN = RGCNConv(self.n_dbpedia, self.dim, self.n_relations, num_bases=self.n_bases)
        self.concept_embeddings = nn.Embedding(self.n_concept, self.dim)
        # rec_loss部分的参数
        self.config.hidden_size=self.dim
        self.con_attn = GPT2Block(self.config)
        self.db_attn = GPT2Block(self.config)
        self.gate_fc = nn.Linear(self.dim * 2, 1)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        # info_loss部分的参数
        self.mse_loss = nn.MSELoss(size_average=False, reduce=False)
        # rating_loss部分的参数
        self.user_embeddings = nn.Embedding(self.n_user, self.embed_dim)
        self.movie_embeddings = nn.Embedding(self.n_dbpedia, self.embed_dim)
        # gen_loss部分的参数
        self.con_fc = nn.Linear(self.dim, self.embed_dim)
        self.db_fc = nn.Linear(self.dim, self.embed_dim)
        self.config.hidden_size=self.embed_dim
        self.encoder = nn.ModuleList([GPT2Block(self.config,layer_idx=i) for i in range(3)])
        self.gen_fc = nn.Linear(self.embed_dim, self.vocab_size)

    @classmethod
    def from_pretrained(cls, args, **kwargs):
        model = super().from_pretrained(args.model_path, **kwargs).to(args.device)
        model.concept_GCN = GCNConv(model.dim, model.dim).to(args.device)
        model.dbpedia_edge_idx = args.dbpedia_edge_list[:, :2].t().to(args.device)
        model.dbpedia_edge_type = args.dbpedia_edge_list[:, 2].to(args.device)
        model.concept_edge_sets = args.concept_edge_sets
        for name, param in model.named_parameters():
            print(f"Module: {name}, Parameters: {param.numel()}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in the model: {total_params}")
        return model

    def forward(self, userIdx, movieId, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector, ignore_index=-100):
        self.batch_size = context_vector.shape[0]
        db_nodes_features = self.dbpedia_RGCN(None, self.dbpedia_edge_idx, self.dbpedia_edge_type)
        con_nodes_features = self.concept_GCN(self.concept_embeddings.weight, self.concept_edge_sets)
        graph_con = con_nodes_features[concept_mask]
        graph_db = db_nodes_features[dbpedia_mask]
        concept_mask = (concept_mask!=-1).view(self.batch_size, -1)
        concept_mask = concept_mask[:, None, None, :]
        concept_mask = concept_mask.to(dtype=self.dtype)  # fp16 compatibility
        concept_mask = (1.0 - concept_mask) * -10000.0
        con_attn = torch.mean(self.con_attn(graph_con, attention_mask=concept_mask)[0], dim=1)
        dbpedia_mask = (dbpedia_mask!=-1).view(self.batch_size, -1)
        dbpedia_mask = dbpedia_mask[:, None, None, :]
        dbpedia_mask = dbpedia_mask.to(dtype=self.dtype)  # fp16 compatibility
        dbpedia_mask = (1.0 - dbpedia_mask) * -10000.0
        db_attn = torch.mean(self.con_attn(graph_db, attention_mask=dbpedia_mask)[0], dim=1)
        # 通过user_emb和db_nodes_features计算rec_scores，对比labels得到rec_loss
        uc_gate = F.sigmoid(self.gate_fc(torch.cat([con_attn, db_attn], dim=-1)))
        user_attn = uc_gate * db_attn + (1 - uc_gate) * con_attn
        rec_scores = F.linear(user_attn, db_nodes_features)
        rec_loss = self.criterion(rec_scores.squeeze(1).squeeze(1).float(), movieId)
        rec_loss = torch.sum(rec_loss.float().to(self.device))
        # 通过db_attn和db_nodes_features计算db_scores，对比db_label得到info_loss
        con_scores = F.linear(db_attn, con_nodes_features)
        db_scores = F.linear(con_attn, db_nodes_features)
        info_db_loss = torch.mean(torch.sum(self.mse_loss(db_scores, dbpedia_vector.float()), dim=-1).float())
        info_con_loss = torch.mean(torch.sum(self.mse_loss(con_scores, concept_vector.float()), dim=-1).float())
        # 计算rating_loss--------------------------------------------------------------
        user_emb = self.user_embeddings(userIdx)  # (batch_size, emsize)
        movie_emb = self.movie_embeddings(movieId)  # (batch_size, emsize)
        ratings = torch.sum(user_emb * movie_emb, 1)  # (batch_size,)
        rating_loss = torch.mean(torch.sum(self.mse_loss(ratings, movie_rating.float()), dim=-1).float())
        # 计算gen_scores和preds---------------------------------------------------------------------------------------------------
        con_emb = self.con_fc(con_attn)
        db_emb = self.db_fc(db_attn)
        context_emb = self.wte(context_vector)  # (batch_size, tgt_len, emsize)
        input_emb = torch.cat([user_emb.unsqueeze(1), movie_emb.unsqueeze(1), context_emb], 1) # (batch_size, total_len, emsize)
        input_mask = torch.cat([torch.ones((self.batch_size, 2), dtype=torch.int64).to(self.device), context_mask], 1)
        position_ids = torch.arange(0, input_emb.shape[1], dtype=torch.long, device=self.device).unsqueeze(0).view(-1, input_emb.shape[1])
        position_emb = self.wpe(position_ids)
        hidden_emb = input_emb + position_emb
        hidden_mask = input_mask.view(self.batch_size, -1)
        hidden_mask = hidden_mask[:, None, None, :]
        hidden_mask = hidden_mask.to(dtype=self.dtype)  # fp16 compatibility
        hidden_mask = (1.0 - hidden_mask) * -10000.0
        for block in self.encoder:
            hidden_emb = block(hidden_emb, attention_mask=hidden_mask)[0]
        response_emb = self.wte(response_vector)  # (batch_size, tgt_len, emsize)
        target_emb = torch.cat([user_emb.unsqueeze(1), movie_emb.unsqueeze(1), con_emb.unsqueeze(1), db_emb.unsqueeze(1), response_emb], 1)
        target_mask = torch.cat([torch.ones((self.batch_size, 4), dtype=torch.int64).to(self.device), response_mask], 1)
        latent = super().forward(attention_mask=target_mask, inputs_embeds=target_emb, encoder_hidden_states=hidden_emb, encoder_attention_mask=input_mask)[0][:,4:,:]
        gen_scores = self.gen_fc(latent)
        gen_loss = torch.mean(self.criterion(gen_scores.view(-1, gen_scores.size(-1)).to(self.device), response_vector.view(-1).to(self.device)))
        _, preds = gen_scores.max(dim=2)
        return preds, rec_scores, gen_loss, rec_loss, rating_loss, info_db_loss, info_con_loss

    def freeze_llm(self, freezeLLM):
        if freezeLLM:
            for block in self.h:
                for param in block.attn.parameters():
                    param.requires_grad = False
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.wpe.parameters():
                param.requires_grad = False
            total_params = sum(p.numel() for p in self.h.parameters())
            total_params += sum(p.numel() for p in self.wte.parameters())
            total_params += sum(p.numel() for p in self.wpe.parameters())
            print(f"Freeze parameters in the model: {total_params}")
        else:
            for block in self.h:
                for param in block.attn.parameters():
                    param.requires_grad = True
            for param in self.wte.parameters():
                param.requires_grad = True
            for param in self.wpe.parameters():
                param.requires_grad = True

    def save_model(self):
        torch.save(self.state_dict(), 'net_parameter.pkl')

    def load_model(self):
        self.load_state_dict(torch.load('net_parameter.pkl'))  





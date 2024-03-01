import torch
import json, math
import pickle as pkl
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from crsdataset import CRSDataset
from crsmodel import Bert4KGModel
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore")


class TrainLoop:
    def __init__(self):
        self.crs_data_path = "data_crs"
        self.batch_size = 64
        self.learning_rate = 0.001
        self.gradient_clip = 0.1
        self.optimizer = 'adam'
        self.device = 'cpu'
        self.n_user = 1075
        self.n_concept = 29308
        self.n_mood = 10
        self.n_dbpedia = 64363
        self.n_relations = 64  # 46+18
        self.n_bases = 8
        self.hidden_dim = 128
        self.max_c_length = 256
        self.max_r_length = 30
        self.embedding_size = 300
        self.n_heads = 2
        self.n_layers = 2
        self.ffn_size = 300
        self.dropout = 0.1
        self.attention_dropout = 0.0
        self.relu_dropout = 0.1
        self.encoder_output_pooling = 'first'
        self.movieIds = pkl.load(open(self.crs_data_path + "/redial_movieIds.pkl", "rb"))
        self.movieId2movie = json.load(open(self.crs_data_path + '/redial_movieId2movie.jsonl', encoding='utf-8'))
        self.text2movie = pkl.load(open(self.crs_data_path + '/redial_text2movie.pkl', 'rb'))
        self.userId2userIdx = json.load(open(self.crs_data_path + '/redial_userId2userIdx.jsonl', encoding='utf-8'))
        self.movie2dbpediaId = pkl.load(open(self.crs_data_path + '/dbpedia_movie2dbpediaId.pkl', 'rb'))
        self.concept2conceptIdx = json.load(open(self.crs_data_path + '/concept_concept2conceptIdx.jsonl', encoding='utf-8'))
        self.concept_edges = open(self.crs_data_path + '/concept_edges.txt', encoding='utf-8')
        self.stopwords = set([word.strip() for word in open(self.crs_data_path + '/stopwords.txt', encoding='utf-8')])
        self.word2wordIdx = json.load(open(self.crs_data_path + '/redial_word2wordIdx.jsonl', encoding='utf-8'))
        # self.wordIdx2word = json.load(open(self.crs_data_path + '/redial_wordIdx2word.jsonl', encoding='utf-8'))
        self.dbpedia_subkg = json.load(open(self.crs_data_path + '/dbpedia_subkg.jsonl', encoding='utf-8'))
        self.wordIdx2word = {self.word2wordIdx[key]: key for key in self.word2wordIdx}
        self.word2wordEmb = np.load(self.crs_data_path + '/redial_word2wordEmb.npy')
        self.special_wordIdx = {'<pad>': 0, '<dbpedia>': 1, '<concept>': 2, '<unk>': 3, '<split>': 4, '<user>': 5, '<movie>': 6, '<mood>': 7, '<eos>': 8, '<related>': 9, '<relation>': 10}
        self.vocab_size = len(self.word2wordIdx) + len(self.special_wordIdx)
        self.train_dataset = CRSDataset('toy_train', self)
        self.valid_dataset = CRSDataset('toy_valid', self)
        self.test_dataset = CRSDataset('toy_test', self)
        # self.train_dataset = CRSDataset('train', self)
        # self.valid_dataset = CRSDataset('valid', self)
        # self.test_dataset = CRSDataset('test', self)
        self.dbpedia_edge_list = self.train_dataset.dbpedia_edge_list.to(self.device)
        self.concept_edge_sets = self.train_dataset.concept_edge_sets.to(self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.metrics_rec = {"rec_loss": 0, "recall@1": 0, "recall@10": 0, "recall@50": 0, "count": 0}
        self.metrics_gen = {"gen_loss": 0, "ppl": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0, "dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0, "count": 0}
        self.model = Bert4KGModel(self).to(self.device)
        self.optimizer = {k.lower(): v for k, v in torch.optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}[self.optimizer]([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate, amsgrad=True, betas=(0.9, 0.999))

    def train(self, rec_epoch, gen_epoch):
        best_val = 10000
        for i in range(rec_epoch + gen_epoch):
            self.model.train()
            if i >= rec_epoch:
                self.model.freeze_kg(True)
            losses = []
            bare_num = 0
            bare_value = 10000
            for num, (userIdx, dbpediaId, context_vector, context_mask, context_pos, context_vm, concept_mentioned, dbpedia_mentioned, related_mentioned, relation_mentioned, user_mentioned, response_vector, response_mask, response_pos, response_vm, concept_vector, dbpedia_vector) in enumerate(tqdm(self.train_dataloader)):
                self.optimizer.zero_grad()
                info_loss, rec_scores, rec_loss, rec2_scores, rec2_loss, predict_vector, gen_loss = self.model(userIdx.to(self.device), dbpediaId.to(self.device), context_vector.to(self.device), context_mask.to(self.device), context_pos.to(self.device), context_vm.to(self.device), concept_mentioned.to(self.device), dbpedia_mentioned.to(self.device), related_mentioned.to(self.device), relation_mentioned.to(self.device), user_mentioned.to(self.device), response_vector.to(self.device), response_mask.to(self.device), response_pos.to(self.device), response_vm.to(self.device), concept_vector.to(self.device), dbpedia_vector.to(self.device))
                if i < rec_epoch:
                    joint_loss = rec_loss + info_loss
                else:
                    joint_loss = gen_loss + rec2_loss
                if joint_loss > bare_value:
                    bare_num += 1
                else:
                    bare_num = 0
                if bare_num > 3:
                    continue
                joint_loss.backward()
                self.optimizer.step()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                losses.append([rec_loss.item(), info_loss.item(), gen_loss.item(), rec2_loss.item(), joint_loss.item()])
                if (num + 1) % (1024 / self.batch_size) == 0:
                    print('epoch%d num%d' % (i, num + 1))
                    print('rec_loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                    print('info_loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
                    print('gen_loss is %f' % (sum([l[2] for l in losses]) / len(losses)))
                    print('rec2_loss is %f' % (sum([l[3] for l in losses]) / len(losses)))
                    print('joint_loss is %f' % (sum([l[4] for l in losses]) / len(losses)))
                    losses = []
            output_metrics_rec, output_metrics_gen = self.val()
            if i < rec_epoch:
                if best_val < output_metrics_rec["rec_loss"]:
                    break
                else:
                    best_val = output_metrics_rec["rec_loss"]
                    self.model.save_model('rec')
                    print("recommendation model saved once------------------------------------------------")
            elif i == rec_epoch:
                best_val = output_metrics_gen["gen_loss"]
                self.model.save_model('gen')
                print("generator model saved once------------------------------------------------")
            else:
                if best_val < output_metrics_gen["gen_loss"]:
                    break
                else:
                    best_val = output_metrics_gen["gen_loss"]
                    self.model.save_model('gen')
                    print("generator model saved once------------------------------------------------")

    def val(self, is_test=False):
        self.model.eval()
        self.metrics_rec = {"rec_loss": 0, "recall@1": 0, "recall@10": 0, "recall@50": 0, "rec2_loss": 0, "recall2@1": 0, "recall2@10": 0, "recall2@50": 0, "count": 0}
        self.metrics_gen = {"gen_loss": 0, "ppl": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0, "dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0, "count": 0}

        def vector2sentence(batch_sen):
            sentences = []
            for sen in batch_sen.numpy().tolist():
                sentence = []
                for word in sen:
                    if word == self.special_wordIdx['<unk>']:
                        sentence.append('_UNK_')
                    if word == self.special_wordIdx['<dbpedia>']:
                        sentence.append('_DBPEDIA_')
                    elif word >= len(self.special_wordIdx):
                        sentence.append(self.wordIdx2word[word])
                sentences.append(sentence)
            return sentences

        val_dataloader = self.test_dataloader if is_test else self.valid_dataloader
        tokens_response = []
        tokens_predict = []
        tokens_context = []
        for userIdx, dbpediaId, context_vector, context_mask, context_pos, context_vm, concept_mentioned, dbpedia_mentioned, related_mentioned, relation_mentioned, user_mentioned, response_vector, response_mask, response_pos, response_vm, concept_vector, dbpedia_vector in tqdm(val_dataloader):
            with torch.no_grad():
                _, rec_scores, rec_loss, rec2_scores, rec2_loss, _, gen_loss = self.model(userIdx.to(self.device), dbpediaId.to(self.device), context_vector.to(self.device), context_mask.to(self.device), context_pos.to(self.device), context_vm.to(self.device), concept_mentioned.to(self.device), dbpedia_mentioned.to(self.device), related_mentioned.to(self.device), relation_mentioned.to(self.device), user_mentioned.to(self.device), response_vector.to(self.device), response_mask.to(self.device), response_pos.to(self.device), response_vm.to(self.device), concept_vector.to(self.device), dbpedia_vector.to(self.device))
                _, _, _, _, _, predict_vector, _ = self.model(userIdx.to(self.device), dbpediaId.to(self.device), context_vector.to(self.device), context_mask.to(self.device), context_pos.to(self.device), context_vm.to(self.device), concept_mentioned.to(self.device), dbpedia_mentioned.to(self.device), related_mentioned.to(self.device), relation_mentioned.to(self.device), user_mentioned.to(self.device), None, None, None, None, concept_vector.to(self.device), dbpedia_vector.to(self.device))
            self.metrics_rec["rec_loss"] += rec_loss.item()
            self.metrics_rec["rec2_loss"] += rec2_loss.item()
            self.metrics_gen['gen_loss'] += gen_loss.item()
            _, pred_idx = torch.topk(rec_scores.cpu()[:, torch.LongTensor(self.movieIds)], k=100, dim=1)
            _, pred2_idx = torch.topk(rec2_scores.cpu()[:, torch.LongTensor(self.movieIds)], k=100, dim=1)
            for b in range(context_vector.shape[0]):
                if dbpediaId[b].item() == 0:
                    continue
                target_idx = self.movieIds.index(dbpediaId[b].item())
                self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
                self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
                self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
                self.metrics_rec["recall2@1"] += int(target_idx in pred2_idx[b][:1].tolist())
                self.metrics_rec["recall2@10"] += int(target_idx in pred2_idx[b][:10].tolist())
                self.metrics_rec["recall2@50"] += int(target_idx in pred2_idx[b][:50].tolist())
                self.metrics_rec["count"] += 1
            print(vector2sentence(predict_vector.cpu()))
            tokens_response.extend(vector2sentence(response_vector.cpu()))
            tokens_predict.extend(vector2sentence(predict_vector.cpu()))
            tokens_context.extend(vector2sentence(context_vector.cpu()))
        for out, tar in zip(tokens_predict, tokens_response):
            self.metrics_gen['bleu1'] += sentence_bleu([tar], out, weights=(1, 0, 0, 0))
            self.metrics_gen['bleu2'] += sentence_bleu([tar], out, weights=(0, 1, 0, 0))
            self.metrics_gen['bleu3'] += sentence_bleu([tar], out, weights=(0, 0, 1, 0))
            self.metrics_gen['bleu4'] += sentence_bleu([tar], out, weights=(0, 0, 0, 1))
            self.metrics_gen['count'] += 1
        unigram_count = 0
        bigram_count = 0
        trigram_count = 0
        quagram_count = 0
        unigram_set = set()
        bigram_set = set()
        trigram_set = set()
        quagram_set = set()
        # outputs is a list which contains several sentences, each sentence contains several words
        for sen in tokens_predict:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen) - 2):
                trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(len(sen) - 3):
                quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_count += 1
                quagram_set.add(quag)
        self.metrics_gen['dist1'] = len(unigram_set) / len(tokens_predict)  # unigram_count
        self.metrics_gen['dist2'] = len(bigram_set) / len(tokens_predict)  # bigram_count
        self.metrics_gen['dist3'] = len(trigram_set) / len(tokens_predict)  # trigram_count
        self.metrics_gen['dist4'] = len(quagram_set) / len(tokens_predict)  # quagram_count
        text_response = [' '.join(tokens) for tokens in tokens_response]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        text_context = [' '.join(tokens) for tokens in tokens_context]
        with open('test_gen.txt', 'w', encoding='utf-8') as file:
            for context, predict, response in zip(text_context, text_predict, text_response):
                file.writelines('=' * 100 + '\n')
                file.writelines("context:" + context + '\n')
                file.writelines("response:" + response + '\n')
                file.writelines("predict:" + predict + '\n')
        self.metrics_rec = {key: self.metrics_rec[key] / self.metrics_gen['count'] if 'loss' in key else self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        self.metrics_gen = {key: self.metrics_gen[key] if 'dist' in key else self.metrics_gen[key] / self.metrics_gen['count'] for key in self.metrics_gen}
        self.metrics_gen['ppl'] = math.exp(self.metrics_gen['gen_loss'])
        print(self.metrics_rec)
        print(self.metrics_gen)
        return self.metrics_rec, self.metrics_gen


if __name__ == '__main__':
    loop = TrainLoop()
    # loop.model.load_model('rec')
    loop.train(rec_epoch=1, gen_epoch=1)
    met = loop.val(is_test=True)

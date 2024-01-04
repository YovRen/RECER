from cgi import test
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import json,math
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from crsdataset import CRSDataset, concept_edge_list, dbpedia_edge_list
from GPT24KG import GPT24KG
from transformers import GPT2Tokenizer
from utils import bleu_score, ids2tokens, unique_sentence_percent
import warnings
warnings.filterwarnings("ignore")


class TrainLoop():
    def __init__(self):
        self.crs_data_path = "data_crs"
        self.model_path = "gpt2"
        self.batch_size = 2
        self.learningrate = 0.001
        self.gradient_clip=0.1
        self.optimizer = 'adam'
        self.device = 'cpu'
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_user = 1075
        self.n_concept = 29308
        self.n_dbpedia = 64363
        self.n_relations = 46
        self.n_bases = 8
        self.dim = 120

        self.max_c_length = 80
        self.max_r_length = 20

        self.movieId2movie = json.load(open(self.crs_data_path + '/redial_movieId2movie.jsonl', encoding='utf-8'))
        self.userId2userIdx = json.load(open(self.crs_data_path + '/redial_userId2userIdx.jsonl', encoding='utf-8'))
        self.concept2conceptIdx = json.load(open(self.crs_data_path + '/concept_concept2conceptIdx.jsonl', encoding='utf-8'))
        self.movie2dbpediaId = pkl.load(open(self.crs_data_path + '/dbpedia_movie2dbpediaId.pkl', 'rb'))
        self.movie2dbpediaId[None]=-1
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path, pad_token = "<|endoftext|>")
        self.dbpedia_edge_list = dbpedia_edge_list(self)
        self.concept_edge_sets = concept_edge_list(self)

        self.train_dataset = CRSDataset('draft', self)
        self.test_dataset = CRSDataset('draft', self)
        self.valid_dataset = CRSDataset('draft', self)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.metrics_rec={"rec_loss":0,"recall@1":0,"recall@10":0,"recall@50":0}
        self.metrics_gen={"gen_loss":0,"ppl":0,"bleu1":0,"bleu4":0,"usr":0,"usn":0}
        self.model = GPT24KG.from_pretrained(self)
        self.optimizer = {k.lower(): v for k, v in torch.optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}[self.optimizer]([p for p in self.model.parameters() if p.requires_grad], lr=self.learningrate,amsgrad=True,betas=(0.9,0.999))

    def train(self, epoch, freeze):
        self.model.freeze_llm(freeze)
        joint_losses=[]
        best_val=0
        for i in range(epoch):
            self.model.train()
            for num, (userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector) in enumerate(tqdm(self.train_dataloader)):
                self.optimizer.zero_grad()
                preds, rec_scores, gen_loss, rec_loss, rating_loss, info_db_loss, info_con_loss=self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
                if freeze and i==0:
                    joint_loss=info_db_loss
                elif freeze:
                    joint_loss=rec_loss
                else:
                    joint_loss=gen_loss+0.5*rating_loss
                joint_losses.append([gen_loss, rec_loss,rating_loss,info_db_loss,info_con_loss,joint_loss])
                joint_loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                if num%30==0:
                    print('gen_loss is %f'%(sum([l[0] for l in joint_losses])/len(joint_losses)))
                    print('rec_loss is %f'%(sum([l[1] for l in joint_losses])/len(joint_losses)))
                    print('rating_loss is %f'%(sum([l[2] for l in joint_losses])/len(joint_losses)))
                    print('info_db_loss is %f'%(sum([l[3] for l in joint_losses])/len(joint_losses)))
                    print('info_con_loss is %f'%(sum([l[4] for l in joint_losses])/len(joint_losses)))
                    print('joint_loss is %f'%(sum([l[5] for l in joint_losses])/len(joint_losses)))
                    joint_losses=[]
            if freeze:
                output_metrics_rec = self.val_rec()
                if best_val > output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]:
                    break
                else:
                    best_val = output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]
                    self.model.save_model()
                    print("recommendation model saved once------------------------------------------------")
            else:
                output_metrics_gen = self.val_gen()
                if best_val < output_metrics_gen["ppl"]:
                    pass
                else:
                    best_val = output_metrics_gen["ppl"]
                    self.model.save_model()
                    print("generator model saved once------------------------------------------------")
                
    def val_rec(self,is_test=False):
        self.model.eval()
        val_dataloader = self.test_dataloader if is_test else self.valid_dataloader
        self.metrics_gen['rec_loss'] = 0.0
        self.metrics_rec["recall@1"] = 0
        self.metrics_rec["recall@10"] = 0
        self.metrics_rec["recall@50"] = 0
        for userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector in tqdm(val_dataloader):
            preds, rec_scores, gen_loss, rec_loss, rating_loss, info_db_loss, info_con_loss=self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
            self.metrics_rec["rec_loss"] += rec_loss.item()
            _, pred_idx = torch.topk(rec_scores.cpu(), k=100, dim=1)
            for b in range(self.batch_size):
                self.metrics_rec["recall@1"] += int(movieIdx[b] in pred_idx[b][:1].tolist())
                self.metrics_rec["recall@10"] += int(movieIdx[b] in pred_idx[b][:10].tolist())
                self.metrics_rec["recall@50"] += int(movieIdx[b] in pred_idx[b][:50].tolist())
        self.metrics_rec={key: self.metrics_rec[key] / (self.batch_size*len(val_dataloader)) for key in self.metrics_rec}
        print(self.metrics_rec)
        return self.metrics_rec

    def val_gen(self,is_test=False):
        self.model.eval()
        val_dataloader = self.test_dataloader if is_test else self.valid_dataloader
        context_list=[]
        response_list=[]
        predict_list=[]
        self.metrics_gen['gen_loss'] = 0.0
        for userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector in tqdm(val_dataloader):
            preds, rec_scores, gen_loss, rec_loss, rating_loss, info_db_loss, info_con_loss=self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
            self.metrics_gen['gen_loss'] += gen_loss.item()
            self.metrics_gen['ppl'] = math.exp(self.metrics_gen['gen_loss'])
            if test==True:
                response_list.extend(response_vector.tolist())
                context_list.extend(context_vector.tolist())
                context_vector = context_vector[:, :1].to(self.device)
                context_mask = None
                response_vector = None
                response_mask = None
                for idx in range(self.max_r_length):
                    preds, rec_scores, gen_loss, rec_loss, rating_loss, info_db_loss, info_con_loss=self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
                    last_token = preds[:, -1]
                    context_vector = torch.cat([context_vector, last_token], 1)
                predict_list.extend(context_vector.tolist())
        if test==True:
            tokens_response = [ids2tokens(ids, self.tokenizer) for ids in response_list]
            tokens_predict = [ids2tokens(ids, self.tokenizer) for ids in predict_list]
            tokens_context = [ids2tokens(ids, self.tokenizer) for ids in context_list]
            self.metrics_gen['bleu1'] = bleu_score(tokens_response, tokens_predict, n_gram=1, smooth=False)
            self.metrics_gen['bleu4'] = bleu_score(tokens_response, tokens_predict, n_gram=4, smooth=False)
            self.metrics_gen['usr'], self.metrics_gen['usn'] = unique_sentence_percent(tokens_predict)
            text_response = [' '.join(tokens) for tokens in tokens_response]
            text_predict = [' '.join(tokens) for tokens in tokens_predict]
            text_context = [' '.join(tokens) for tokens in tokens_context]
            with open('test_gen.txt','w',encoding='utf-8') as file:
                for context,predict,response in zip(text_context,text_predict,text_response):
                    file.writelines('='*100+'\n')
                    file.writelines("context:"+context+'\n')
                    file.writelines("response:"+response+'\n')
                    file.writelines("predict:"+predict+'\n')
        print(self.metrics_gen)
        return self.metrics_gen

if __name__ == '__main__':
    loop=TrainLoop()
    loop.train(epoch=3,freeze=True)
    loop.train(epoch=3,freeze=False)
    met=loop.val_rec(is_test = True)
    met=loop.val_gen(is_test = True)
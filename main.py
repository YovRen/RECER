import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from crsdataset import CRSDataset
from GPT24KG import GPT24KG
from utils import rouge_score, bleu_score, now_time, ids2tokens, unique_sentence_percent, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
import warnings
warnings.filterwarnings("ignore")

class Args:
    def __init__(self):
        self.pep_data_path = "data_pep/TripAdvisor/reviews.pickle"
        self.pep_index_dir = "data_pep/TripAdvisor/1/"
        self.crs_data_path = "data_crs"
        self.model_path = "gpt2"

        self.n_user = 1075
        self.n_concept = 29308
        self.n_dbpedia = 64363
        self.n_relations = 46
        self.n_bases = 8

        self.max_c_length = 80
        self.max_r_length = 20
        self.batch_size = 2
        self.epoch = 1
        self.learningrate = 0.001
        self.gradient_clip=0.1
        self.optimizer = 'adam'
        self.dim = 120
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

class TrainLoop():
    def __init__(self, args):
        self.batch_size=args.batch_size
        self.epoch=args.epoch
        self.device=args.device
        self.n_dbpedia=args.n_dbpedia
        self.n_concept=args.n_concept
        self.gradient_clip=args.gradient_clip
        self.optimizer=args.optimizer
        self.learningrate=args.learningrate
        self.train_dataset = CRSDataset('draft', args)
        # self.train_dataset = CRSDataset('valid', args)
        # self.test_dataset = CRSDataset('test', args)
        # self.valid_dataset = CRSDataset('valid', args)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        # self.valid_dataloader = DataLoade
        # r(self.valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.model = GPT24KG.from_pretrained("gpt2").to(self.device)
        self.optimizer = {k.lower(): v for k, v in torch.optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}[self.optimizer]([p for p in self.model.parameters() if p.requires_grad], lr=self.learningrate,amsgrad=True,betas=(0.9,0.999))

    def train(self):
        self.model.train()
        self.model.freeze_llm(True)
        losses=[]
        best_val=0
        for i in range(self.epoch):
            num=0
            for userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                preds, gen_scores, gen_loss, rec_loss, rating_loss, info_db_loss, info_con_loss=self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
                joint_loss=gen_loss+rec_loss+0.025*info_db_loss+0.025*rating_loss
                losses.append([gen_loss, rec_loss,info_db_loss,rating_loss])
                joint_loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                if num%50==0:
                    print('gen_loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('rec_loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    print('info_db_loss is %f'%(sum([l[2] for l in losses])/len(losses)))
                    print('rating_loss is %f'%(sum([l[3] for l in losses])/len(losses)))
                    losses=[]
                num+=1
            print("masked_loss pre-trained------------------------------------------------")
            output_metrics_rec = self.val_rec()
            output_metrics_gen = self.val_gen()
            if best_val > output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]:
                break
            else:
                best_val = output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]
                self.model.save_model()
                print("recommendation model saved once------------------------------------------------")
            if best_val < output_metrics_gen["dist4"]:
                pass
            else:
                best_val = output_metrics_gen["dist4"]
                self.model.save_model()
                print("generator model saved once------------------------------------------------")
                
    def val_rec(self,is_test=False):
        self.model.eval()
        val_dataloader = self.test_dataloader if is_test else self.valid_dataloader
        for userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, movie_vector, dbpedia_vector, concept_vector in tqdm(val_dataloader):
            with torch.no_grad():
                preds, gen_scores, rec_loss, rating_loss, info_db_loss, info_con_loss = self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
            self.metrics_rec["loss"] += rec_loss
            outputs = gen_scores.cpu()
            outputs = outputs[:, torch.LongTensor(self.movie_ids)]
            _, pred_idx = torch.topk(outputs, k=100, dim=1)
            for b in range(self.batch_size):
                if movieIdx[b].item()==0:
                    continue
                target_idx = self.movie_ids.index(movieIdx[b].item())
                self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
                self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
                self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
                self.metrics_rec["count"] += 1
        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        print(output_dict_rec)
        return output_dict_rec

    def val_gen(self,is_test=False):
        self.model.eval()
        val_dataloader = self.test_dataloader if is_test else self.valid_dataloader
        inference_sum=[]
        golden_sum=[]
        context_sum=[]
        losses=[]
        recs=[]
        
        def vector2sentence(batch_sen):
            sentences=[]
            for sen in batch_sen.numpy().tolist():
                sentence=[]
                for word in sen:
                    if word>3:
                        sentence.append(self.index2word[word])
                    elif word==3:
                        sentence.append('_UNK_')
                sentences.append(sentence)
            return sentences

        for userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, movie_vector, dbpedia_vector, concept_vector in tqdm(val_dataloader):
            with torch.no_grad():
                preds, gen_scores, rec_loss, rating_loss, info_db_loss, info_con_loss = self.model(userIdx, movieIdx, movie_rating, context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector)
            golden_sum.extend(vector2sentence(response_vector.cpu()))
            inference_sum.extend(vector2sentence(preds.cpu()))
            context_sum.extend(vector2sentence(context_vector.cpu()))
            losses.append(torch.mean(rec_loss))#TODO
        generated=[]
        for out, tar in zip(inference_sum, golden_sum):
            bleu1 = bleu_score([tar], out, weights=(1, 0, 0, 0))
            bleu2 = bleu_score([tar], out, weights=(0, 1, 0, 0))
            bleu3 = bleu_score([tar], out, weights=(0, 0, 1, 0))
            bleu4 = bleu_score([tar], out, weights=(0, 0, 0, 1))
            generated.append(out)
            self.metrics_gen['bleu1']+=bleu1
            self.metrics_gen['bleu2']+=bleu2
            self.metrics_gen['bleu3']+=bleu3
            self.metrics_gen['bleu4']+=bleu4
            self.metrics_gen['count']+=1
        # outputs is a list which contains several sentences, each sentence contains several words
        unigram_count = 0
        bigram_count = 0
        trigram_count=0
        quagram_count=0
        unigram_set = set()
        bigram_set = set()
        trigram_set=set()
        quagram_set=set()
        for sen in generated:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen)-2):
                trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_count+=1
                trigram_set.add(trg)
            for start in range(len(sen)-3):
                quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_count+=1
                quagram_set.add(quag)
        self.metrics_gen['dist1']=len(unigram_set) / len(generated)#unigram_count
        self.metrics_gen['dist2']=len(bigram_set) / len(generated)#bigram_count
        self.metrics_gen['dist3']=len(trigram_set)/len(generated)#trigram_count
        self.metrics_gen['dist4']=len(quagram_set)/len(generated)#quagram_count
        output_dict_gen={}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['count']
            else:
                output_dict_gen[key]=self.metrics_gen[key]
        f=open('out/context_test.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in context_sum])
        f.close()
        f=open('out/output_test.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
        f.close()
        print(output_dict_gen)
        return output_dict_gen

if __name__ == '__main__':
    args = Args()
    loop=TrainLoop(args)
    loop.train()
    met=loop.val_rec(True)
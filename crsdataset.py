import re
import json
import pickle as pkl
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from transformers import GPT2Tokenizer
from torch.utils.data.dataset import Dataset

class CRSDataset(Dataset):
    def __init__(self, mode, args):
        self.crs_data_path = args.crs_data_path
        self.device = args.device
        self.max_c_length = args.max_c_length
        self.max_r_length = args.max_r_length
        self.movieId2movie = json.load(open(self.crs_data_path + '/redial_movieId2movie.jsonl', encoding='utf-8'))
        self.userId2userIdx = json.load(open(self.crs_data_path + '/redial_userId2userIdx.jsonl', encoding='utf-8'))
        self.concept2conceptIdx = json.load(open(self.crs_data_path + '/concept_concept2conceptIdx.jsonl', encoding='utf-8'))
        self.movie2dbpediaId = pkl.load(open(self.crs_data_path + '/dbpedia_movie2dbpediaId.pkl', 'rb'))
        self.movie2dbpediaId[None]=-1
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_path, pad_token = "<|endoftext|>")
        self.n_dbpedia = len(self.movie2dbpediaId)
        self.n_concept = len(self.concept2conceptIdx)
        self.n_user = len(self.userId2userIdx)
        f = open(self.crs_data_path + '/' + mode + '_data.jsonl', encoding='utf-8')
        self.datapre = []
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            movieMentions = lines['movieMentions']
            respondentWorkerId = lines['respondentWorkerId']
            respondentQuestions = lines['respondentQuestions']
            initiatorQuestions = lines['initiatorQuestions']
            for movieId in movieMentions.keys():
                if len(respondentQuestions)==0:
                    respondentQuestions = dict(respondentQuestions)
                if len(initiatorQuestions)==0:
                    initiatorQuestions = dict(initiatorQuestions)
                if movieId not in respondentQuestions.keys():
                    respondentQuestions[movieId] = {"suggested":0,"seen":2,"liked":2}
                if movieId not in initiatorQuestions.keys():
                    initiatorQuestions[movieId] = {"suggested":0,"seen":2,"liked":2}
            message_list = []
            lastId = -1
            for message in messages:
                for mv in movieMentions.keys():
                    assert self.movieId2movie[mv] in self.movie2dbpediaId.keys()
                digits = re.findall(r'@(\d+)', message['text'])
                processed_text = re.sub(r'#', '!', message['text'])
                
                processed_text = '{} {} {}'.format('<|endoftext|>', processed_text, '<|endoftext|>')
                processed_text = re.sub(r'@(\d+)', '#', processed_text)
                tokens = self.tokenizer.tokenize(processed_text)
                input_ids = self.tokenizer(processed_text)['input_ids']
                sharpIdxs = [i for i in range(len(tokens)) if input_ids[i] == 1303 or input_ids[i] == 2]
                sharpIdx2digit = {sharpIdx:int(digit) for digit,sharpIdx in zip(digits,sharpIdxs)}
                movieIds = [digit for digit in digits if digit.isdigit() and digit in movieMentions.keys() and self.movie2dbpediaId[self.movieId2movie[digit]] != -1]
                movie_ratings = {self.movie2dbpediaId[self.movieId2movie[movieId]]:respondentQuestions[movieId]["suggested"] if message['senderWorkerId']==respondentWorkerId else initiatorQuestions[movieId]["suggested"] for movieId in movieIds}
                concept_mask = [self.concept2conceptIdx.get(token.lower(), -1) for token in tokens]
                dbpedia_mask = [self.movie2dbpediaId[self.movieId2movie[movieIds[sharpIdx2digit[i]]]] if i in sharpIdxs and sharpIdx2digit[i] in movieIds else -1 for i in range(len(tokens))]

                sender_worker_id = message['senderWorkerId']
                if sender_worker_id != lastId:
                    message_dict = {'sender_worker_id': sender_worker_id, 'movie_ratings':movie_ratings, 'input_ids': input_ids,  'concept_mask': concept_mask, 'dbpedia_mask': dbpedia_mask}
                    message_list.append(message_dict)
                    lastId = message['senderWorkerId']
                else:
                    message_list[-1]['movie_ratings'].update(movie_ratings)
                    message_list[-1]['input_ids'] += input_ids
                    message_list[-1]['concept_mask'] += concept_mask
                    message_list[-1]['dbpedia_mask'] += dbpedia_mask

            history_input_ids = []
            history_concept_mask = []
            history_dbpedia_mask = []
            history_movie_ratings = {}
            for message_dict in message_list:

                concept_vector = torch.zeros(self.n_concept, device=self.device)
                for con in history_concept_mask:
                    if con != -1:
                        concept_vector[con] = 1

                dbpedia_vector = torch.zeros(self.n_dbpedia, device=self.device)
                for dbpediaId in history_movie_ratings.keys():
                    dbpedia_vector[dbpediaId] = 1

                userIdx = torch.tensor(int(self.userId2userIdx[str(message_dict['sender_worker_id'])]), dtype=torch.long, device=self.device)
                context_vector = torch.tensor((history_input_ids + [0] * (self.max_c_length - len(history_input_ids)))[:self.max_c_length], dtype=torch.long, device=self.device)
                context_mask = torch.tensor(([1] * len(history_input_ids) + [0] * (self.max_c_length - len(history_input_ids)))[:self.max_c_length], dtype=torch.long, device=self.device)
                concept_mask = torch.tensor((history_concept_mask + [0] * (self.max_c_length - len(history_concept_mask)))[:self.max_c_length], dtype=torch.long, device=self.device)
                dbpedia_mask = torch.tensor((history_dbpedia_mask + [0] * (self.max_c_length - len(history_dbpedia_mask)))[:self.max_c_length], dtype=torch.long, device=self.device)
                response_vector = torch.tensor((message_dict['input_ids'] + [0] * (self.max_r_length - len(message_dict['input_ids'])))[:self.max_r_length], dtype=torch.long, device=self.device)
                response_mask = torch.tensor(([1]*len(message_dict['input_ids']) + [0] * (self.max_r_length - len(message_dict['input_ids'])))[:self.max_r_length], dtype=torch.long, device=self.device)
                assert len(context_vector)==len(context_mask)==len(concept_mask)==len(dbpedia_mask)==self.max_c_length

                self.datapre.extend([[userIdx, torch.tensor(dbpediaId, dtype=torch.long, device=self.device), torch.tensor(message_dict['movie_ratings'][dbpediaId], dtype=torch.long, device=self.device), context_vector, context_mask, response_vector, response_mask, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector] for dbpediaId in message_dict['movie_ratings'].keys()])
                history_input_ids.extend(message_dict['input_ids'])
                history_concept_mask.extend(message_dict['concept_mask'])
                history_dbpedia_mask.extend(message_dict['dbpedia_mask'])
                history_movie_ratings.update(message_dict['movie_ratings'])
            
    def __getitem__(self, index):
        return self.datapre[index]

    def __len__(self):
        return len(self.datapre)


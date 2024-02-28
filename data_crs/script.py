import json
import nltk
import gensim
import numpy as np
from tqdm import tqdm

class DataPre:
    def __init__(self, crs_data_path):
        self.crs_data_path = crs_data_path

    def prepare_dbpedia_user_subkg(self):
        """
        准备 Word2Vec 模型，对语料库进行处理并生成词向量 '<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3, '<split>':4, '<user>':5, '<movie>':6, '<mood>':7
        """
        self.corpus = []
        f = open(self.crs_data_path + '/' + 'train_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            for message in messages:
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                self.corpus.append(tokens)
        f = open(self.crs_data_path + '/' + 'test_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            for message in messages:
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                self.corpus.append(tokens)
        f = open(self.crs_data_path + '/' + 'valid_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            for message in messages:
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                self.corpus.append(tokens)
        model = gensim.models.word2vec.Word2Vec(self.corpus, vector_size=300, min_count=1)
        word2wordIdx = {word: i + len(self.special_wordIdx) for i, word in enumerate(model.wv.index_to_key)}
        wordIdx2word = {i + len(self.special_wordIdx): word for i, word in enumerate(model.wv.index_to_key)}
        word2embedding = [[0] * 300] * len(self.special_wordIdx) + [model.wv[word] for word in word2wordIdx]
        json.dump(word2wordIdx, open(self.crs_data_path + '/redial_word2wordIdx.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(wordIdx2word, open(self.crs_data_path + '/redial_wordIdx2word.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)
        np.save(self.crs_data_path + '/redial_word2wordEmb.npy', word2embedding)

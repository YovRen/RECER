import json
import nltk
import torch
import gensim
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data.dataset import Dataset


class CRSDataset(Dataset):
    def __init__(self, mode, args):
        self.device = args.device
        self.crs_data_path = args.crs_data_path
        self.max_c_length = args.max_c_length
        self.max_r_length = args.max_r_length
        self.movieId2movie = args.movieId2movie
        self.text2movie = args.text2movie
        self.userId2userIdx = args.userId2userIdx
        self.concept2conceptIdx = args.concept2conceptIdx
        self.concept_edges = args.concept_edges
        self.stopwords = args.stopwords
        self.word2wordIdx = args.word2wordIdx
        self.movie2dbpediaId = args.movie2dbpediaId
        self.special_wordIdx = args.special_wordIdx
        self.n_dbpedia = args.n_dbpedia
        self.n_concept = args.n_concept
        self.datapre = []
        # self.prepare_word2vec()
        # self.prepare_dbpedia_subkg()
        self.dbpedia_subkg = args.dbpedia_subkg
        self.dbpedia_edge_list = self.dbpedia_edge_list()
        self.concept_edge_sets = self.concept_edge_sets()
        f = open(self.crs_data_path + '/' + mode + '_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            movieMentions = lines['movieMentions']
            respondentWorkerId = lines['respondentWorkerId']
            respondentQuestions = lines['respondentQuestions']
            initiatorQuestions = lines['initiatorQuestions']
            if len(movieMentions) == 0:
                continue
            if len(respondentQuestions) == 0:
                respondentQuestions = {}
            if len(initiatorQuestions) == 0:
                initiatorQuestions = {}
            for movieId in movieMentions.keys():
                if movieId not in respondentQuestions.keys():
                    respondentQuestions[movieId] = {"suggested": 0, "seen": 1, "liked": 1}
                if movieId not in initiatorQuestions.keys():
                    initiatorQuestions[movieId] = {"suggested": 0, "seen": 1, "liked": 1}
            # 将所有的消息更新到message_list中
            message_list = []
            for num, message in enumerate(messages):
                input_ids = []
                attention_mask = []
                concept_mentioned = []
                dbpedia_mentioned = []
                related_mentioned = []
                relation_mentioned = []
                # create tree
                sent_tree = []
                pos_idx_tree = []
                abs_idx_tree = []
                abs_idx_src = []
                pos_idx = -1
                abs_idx = -1
                # 用于检测消息中提及的电影并转换成特定格式
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                for token in tokens:
                    if token.isdigit() and token in movieMentions.keys() and self.movieId2movie[token] in self.movie2dbpediaId.keys():
                        relates = self.dbpedia_subkg.get(str(self.movie2dbpediaId[self.movieId2movie[token]]), [])
                    elif token.isdigit() and token in self.text2movie[message['text']]:
                        relates = self.dbpedia_subkg.get(str(self.movie2dbpediaId[self.movieId2movie[token]]), [])
                    else:
                        relates = []
                    sent_tree.append((token, relates))
                    pos_idx += 1
                    abs_idx += 1
                    relates_pos_idx = []
                    relates_abs_idx = []
                    last_abs_idx = abs_idx
                    for relate in relates:
                        relates_pos_idx.append([pos_idx + 1, pos_idx + 2])
                        relates_abs_idx.append([abs_idx + 1, abs_idx + 2])
                        abs_idx += 2
                    pos_idx_tree.append((pos_idx, relates_pos_idx))
                    abs_idx_tree.append((last_abs_idx, relates_abs_idx))
                    abs_idx_src.append(last_abs_idx)
                # 根据树构建其他内容input_ids&attention_mask
                for i, (token, relates) in enumerate(sent_tree):
                    if token in self.concept2conceptIdx.keys():
                        input_ids.append(self.word2wordIdx.get(token, self.special_wordIdx['<unk>']))
                        attention_mask.append(self.special_wordIdx['<concept>'])
                        concept_mentioned.append(self.concept2conceptIdx[token])
                    elif token.isdigit() and token in movieMentions.keys() and self.movieId2movie[token] in self.movie2dbpediaId.keys():
                        input_ids.append(self.special_wordIdx['<dbpedia>'])
                        attention_mask.append(self.special_wordIdx['<dbpedia>'])
                        dbpedia_mentioned.append(self.movie2dbpediaId[self.movieId2movie[token]])
                    elif token.isdigit() and token in self.text2movie[message['text']]:
                        input_ids.append(self.special_wordIdx['<dbpedia>'])
                        attention_mask.append(self.special_wordIdx['<dbpedia>'])
                        dbpedia_mentioned.append(self.movie2dbpediaId[self.movieId2movie[token]])
                    else:
                        input_ids.append(self.word2wordIdx.get(token, self.special_wordIdx['<unk>']))
                        attention_mask.append(self.special_wordIdx['<unk>'])
                    for relate in relates:
                        input_ids.append(self.special_wordIdx['<related>'])
                        input_ids.append(self.special_wordIdx['<relation>'])
                        attention_mask += [self.special_wordIdx['<related>'], self.special_wordIdx['<relation>']]
                        related_mentioned.append(int(relate[0]))
                        relation_mentioned.append(int(relate[1]))
                # 合并每条消息
                if num == 0:
                    message_dict = {'userIdx': int(self.userId2userIdx[str(message['senderWorkerId'])]), 'input_ids': input_ids, 'attention_mask': attention_mask, 'concept_mentioned': concept_mentioned, 'dbpedia_mentioned': dbpedia_mentioned, 'related_mentioned': related_mentioned, 'relation_mentioned': relation_mentioned, 'abs_idx_tree': abs_idx_tree, 'pos_idx_tree': pos_idx_tree, 'abs_idx_src': abs_idx_src, 'pos_idx': pos_idx, 'abs_idx': abs_idx}
                    message_list.append(message_dict)
                elif int(self.userId2userIdx[str(message['senderWorkerId'])]) == message_list[-1]['userIdx']:
                    message_list[-1]['concept_mentioned'] += concept_mentioned
                    message_list[-1]['dbpedia_mentioned'] += dbpedia_mentioned
                    message_list[-1]['related_mentioned'] += related_mentioned
                    message_list[-1]['relation_mentioned'] += relation_mentioned
                    message_list[-1]['input_ids'] += input_ids
                    message_list[-1]['attention_mask'] += attention_mask
                    message_list[-1]['pos_idx_tree'] += [(src_id + message_list[-1]['pos_idx'] + 1, [[relate[0] + message_list[-1]['pos_idx'] + 1, relate[1] + message_list[-1]['pos_idx'] + 1] for relate in relates]) for src_id, relates in pos_idx_tree]
                    message_list[-1]['abs_idx_tree'] += [(src_id + message_list[-1]['abs_idx'] + 1, [[relate[0] + message_list[-1]['abs_idx'] + 1, relate[1] + message_list[-1]['abs_idx'] + 1] for relate in relates]) for src_id, relates in abs_idx_tree]
                    message_list[-1]['abs_idx_src'] += [src_id + message_list[-1]['abs_idx'] + 1 for src_id in abs_idx_src]
                    message_list[-1]['pos_idx'] += (pos_idx + 1)
                    message_list[-1]['abs_idx'] += (abs_idx + 1)
                else:
                    message_list[-1]['input_ids'] = [self.special_wordIdx['<user>']] + [self.special_wordIdx['<movie>']] * len(message_list[-1]['dbpedia_mentioned']) + message_list[-1]['input_ids'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['attention_mask'] = [self.special_wordIdx['<user>']] + [self.special_wordIdx['<movie>']] * len(message_list[-1]['dbpedia_mentioned']) + message_list[-1]['attention_mask'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['pos_idx_tree'] = [(0, [])] + [(1, [])] * len(message_list[-1]['dbpedia_mentioned']) + [(src_id + len(message_list[-1]['dbpedia_mentioned']) + 1, [[relate[0] + len(message_list[-1]['dbpedia_mentioned']) + 1, relate[1] + len(message_list[-1]['dbpedia_mentioned']) + 1] for relate in relates]) for src_id, relates in message_list[-1]['pos_idx_tree']] + [(message_list[-1]['pos_idx'] + len(message_list[-1]['dbpedia_mentioned']) + 2, [])]
                    message_list[-1]['abs_idx_tree'] = [(i, []) for i in range(len(message_list[-1]['dbpedia_mentioned']) + 1)] + [(src_id + len(message_list[-1]['dbpedia_mentioned']) + 1, [[relate[0] + len(message_list[-1]['dbpedia_mentioned']) + 1, relate[1] + len(message_list[-1]['dbpedia_mentioned']) + 1] for relate in relates]) for src_id, relates in message_list[-1]['abs_idx_tree']] + [(message_list[-1]['abs_idx'] + len(message_list[-1]['dbpedia_mentioned']) + 2, [])]
                    message_list[-1]['abs_idx_src'] = list(range(len(message_list[-1]['dbpedia_mentioned']) + 1)) + [src_id + len(message_list[-1]['dbpedia_mentioned']) + 1 for src_id in message_list[-1]['abs_idx_src']] + [message_list[-1]['abs_idx'] + len(message_list[-1]['dbpedia_mentioned']) + 2]
                    message_list[-1]['pos_idx'] += (len(message_list[-1]['dbpedia_mentioned']) + 2)
                    message_list[-1]['abs_idx'] += (len(message_list[-1]['dbpedia_mentioned']) + 2)
                    message_dict = {'userIdx': int(self.userId2userIdx[str(message['senderWorkerId'])]), 'input_ids': input_ids, 'attention_mask': attention_mask, 'concept_mentioned': concept_mentioned, 'dbpedia_mentioned': dbpedia_mentioned, 'related_mentioned': related_mentioned, 'relation_mentioned': relation_mentioned, 'abs_idx_tree': abs_idx_tree, 'pos_idx_tree': pos_idx_tree, 'abs_idx_src': abs_idx_src, 'pos_idx': pos_idx, 'abs_idx': abs_idx}
                    message_list.append(message_dict)
                if num == len(messages) - 1:
                    message_list[-1]['input_ids'] = [self.special_wordIdx['<user>']] + [self.special_wordIdx['<movie>']] * len(message_list[-1]['dbpedia_mentioned']) + message_list[-1]['input_ids'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['attention_mask'] = [self.special_wordIdx['<user>']] + [self.special_wordIdx['<movie>']] * len(message_list[-1]['dbpedia_mentioned']) + message_list[-1]['attention_mask'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['pos_idx_tree'] = [(0, [])] + [(1, [])] * len(message_list[-1]['dbpedia_mentioned']) + [(src_id + len(message_list[-1]['dbpedia_mentioned']) + 1, [[relate[0] + len(message_list[-1]['dbpedia_mentioned']) + 1, relate[1] + len(message_list[-1]['dbpedia_mentioned']) + 1] for relate in relates]) for src_id, relates in message_list[-1]['pos_idx_tree']] + [(message_list[-1]['pos_idx'] + len(message_list[-1]['dbpedia_mentioned']) + 2, [])]
                    message_list[-1]['abs_idx_tree'] = [(i, []) for i in range(len(message_list[-1]['dbpedia_mentioned']) + 1)] + [(src_id + len(message_list[-1]['dbpedia_mentioned']) + 1, [[relate[0] + len(message_list[-1]['dbpedia_mentioned']) + 1, relate[1] + len(message_list[-1]['dbpedia_mentioned']) + 1] for relate in relates]) for src_id, relates in message_list[-1]['abs_idx_tree']] + [(message_list[-1]['abs_idx'] + len(message_list[-1]['dbpedia_mentioned']) + 2, [])]
                    message_list[-1]['abs_idx_src'] = list(range(len(message_list[-1]['dbpedia_mentioned']) + 1)) + [src_id + len(message_list[-1]['dbpedia_mentioned']) + 1 for src_id in message_list[-1]['abs_idx_src']] + [message_list[-1]['abs_idx'] + len(message_list[-1]['dbpedia_mentioned']) + 2]
                    message_list[-1]['pos_idx'] += (3 if len(message_list[-1]['dbpedia_mentioned']) > 0 else 2)
                    message_list[-1]['abs_idx'] += (len(message_list[-1]['dbpedia_mentioned']) + 2)

            # 将message_list中所有的消息整合起来
            history_concept_mentioned = []
            history_dbpedia_mentioned = []
            history_related_mentioned = []
            history_relation_mentioned = []
            history_user_mentioned = []
            history_input_ids = []
            history_attention_mask = []
            history_pos_idx_tree = []
            history_abs_idx_tree = []
            history_abs_idx_src = []
            history_abs_idx = -1
            history_pos_idx = -1
            for message_dict in message_list:
                userIdx = message_dict['userIdx']
                if userIdx == int(self.userId2userIdx[str(respondentWorkerId)]) and len(history_input_ids) > 0:
                    context_vector = np.array((history_input_ids + [self.special_wordIdx['<eos>']] + [self.special_wordIdx['<pad>']] * (self.max_c_length - len(history_input_ids) - 1))[:self.max_c_length])
                    context_mask = np.array((history_attention_mask + [self.special_wordIdx['<eos>']] + [self.special_wordIdx['<pad>']] * (self.max_c_length - len(history_attention_mask) - 1))[:self.max_c_length])
                    concept_mentioned = np.array((history_concept_mentioned + [0] * (self.max_c_length - len(history_concept_mentioned)))[:self.max_c_length])
                    dbpedia_mentioned = np.array((history_dbpedia_mentioned + [0] * (50 - len(history_dbpedia_mentioned)))[:50])
                    related_mentioned = np.array((history_related_mentioned + [0] * (200 - len(history_related_mentioned)))[:200])
                    relation_mentioned = np.array((history_relation_mentioned + [0] * (200 - len(history_relation_mentioned)))[:200])
                    user_mentioned = np.array((history_user_mentioned + [0] * (50 - len(history_user_mentioned)))[:50])
                    response_vector = np.array((message_dict['input_ids'] + [self.special_wordIdx['<eos>']] + [self.special_wordIdx['<pad>']] * (self.max_r_length - len(message_dict['input_ids']) - 1))[:self.max_r_length])
                    response_mask = np.array((message_dict['attention_mask'] + [self.special_wordIdx['<eos>']] + [self.special_wordIdx['<pad>']] * (self.max_r_length - len(message_dict['attention_mask']) - 1))[:self.max_r_length])
                    # Calculate visible matrix
                    context_pos = []
                    for src_id, relates in history_pos_idx_tree:
                        context_pos.append(src_id)
                        for relate in relates:
                            context_pos += relate
                    context_pos = np.array((context_pos + [self.max_c_length - 1] * (self.max_c_length - len(context_pos)))[:self.max_c_length])
                    response_pos = []
                    for src_id, relates in message_dict['pos_idx_tree']:
                        response_pos.append(src_id)
                        for relate in relates:
                            response_pos += relate
                    response_pos = np.array((response_pos + [self.max_r_length - 1] * (self.max_r_length - len(response_pos)))[:self.max_r_length])
                    # 构建可视矩阵
                    context_vm = np.zeros((history_abs_idx + 1, history_abs_idx + 1))
                    history_dbpedia_count = []
                    for src_id, relates in history_abs_idx_tree:
                        context_vm[src_id, history_abs_idx_src + [idx for relate in relates for idx in relate]] = 1
                        for relate in relates:
                            history_dbpedia_count_len = len(history_dbpedia_count)
                            for i, (epoch_src_id, epoch_relates) in enumerate(history_dbpedia_count):
                                for epoch_relate in epoch_relates:
                                    context_vm[relate[0], epoch_relate + [epoch_src_id]] = 0.9**(history_dbpedia_count_len-i)
                                    context_vm[relate[1], epoch_relate + [epoch_src_id]] = 0.9**(history_dbpedia_count_len-i)
                            context_vm[relate[0], relate + [src_id]] = 1
                            context_vm[relate[1], relate + [src_id]] = 1
                        if len(relates)>0:
                            history_dbpedia_count.append((src_id, relates))
                    if history_abs_idx < self.max_c_length:
                        context_vm = np.pad(context_vm, ((0, self.max_c_length - history_abs_idx - 1), (0, self.max_c_length - history_abs_idx - 1)), 'constant')
                    else:
                        context_vm = context_vm[:self.max_c_length, :self.max_c_length]
                    response_vm = np.zeros((message_dict['abs_idx'] + 1, message_dict['abs_idx'] + 1))
                    history_dbpedia_count = []
                    for src_id, relates in message_dict['abs_idx_tree']:
                        response_vm[src_id, message_dict['abs_idx_src'] + [idx for relate in relates for idx in relate]] = 1
                        for relate in relates:
                            history_dbpedia_count_len = len(history_dbpedia_count)
                            for i, (epoch_src_id, epoch_relates) in enumerate(history_dbpedia_count):
                                for epoch_relate in epoch_relates:
                                    response_vm[relate[0], epoch_relate + [epoch_src_id]] = 0.9 ** (history_dbpedia_count_len - i)
                                    response_vm[relate[1], epoch_relate + [epoch_src_id]] = 0.9 ** (history_dbpedia_count_len - i)
                            response_vm[relate[0], relate + [src_id]] = 1
                            response_vm[relate[1], relate + [src_id]] = 1
                        if len(relates) > 0:
                            history_dbpedia_count.append((src_id, relates))
                    if message_dict['abs_idx'] < self.max_r_length:
                        response_vm = np.pad(response_vm, ((0, self.max_r_length - message_dict['abs_idx'] - 1), (0, self.max_r_length - message_dict['abs_idx'] - 1)), 'constant')
                    else:
                        response_vm = response_vm[:self.max_r_length, :self.max_r_length]

                    if len(message_dict['dbpedia_mentioned']) == 0:
                        self.datapre.append([userIdx, 0, context_vector, context_mask, context_pos, context_vm, concept_mentioned, dbpedia_mentioned, related_mentioned, relation_mentioned, user_mentioned, response_vector, response_mask, response_pos, response_vm])
                    for dbpediaId in message_dict['dbpedia_mentioned']:
                        self.datapre.append([userIdx, dbpediaId, context_vector, context_mask, context_pos, context_vm, concept_mentioned, dbpedia_mentioned, related_mentioned, relation_mentioned, user_mentioned, response_vector, response_mask, response_pos, response_vm])
                if len(history_attention_mask) != 0:
                    history_input_ids.extend([self.special_wordIdx['<split>']])
                    history_attention_mask.extend([self.special_wordIdx['<split>']])
                    history_pos_idx_tree.extend([(history_pos_idx + 1, [])])
                    history_abs_idx_tree.extend([(history_abs_idx + 1, [])])
                    history_abs_idx_src.extend([history_abs_idx + 1])
                    history_abs_idx += 1
                    history_pos_idx += 1
                history_concept_mentioned.extend(message_dict['concept_mentioned'])
                history_dbpedia_mentioned.extend(message_dict['dbpedia_mentioned'])
                history_related_mentioned.extend(message_dict['related_mentioned'])
                history_relation_mentioned.extend(message_dict['relation_mentioned'])
                history_user_mentioned.extend([message_dict['userIdx']])
                history_input_ids.extend(message_dict['input_ids'])
                history_attention_mask.extend(message_dict['attention_mask'])
                history_pos_idx_tree.extend([(src_id + history_pos_idx + 1, [[relate[0] + history_pos_idx + 1, relate[1] + history_pos_idx + 1] for relate in relates]) for src_id, relates in message_dict['pos_idx_tree']])
                history_abs_idx_tree.extend([(src_id + history_abs_idx + 1, [[relate[0] + history_abs_idx + 1, relate[1] + history_abs_idx + 1] for relate in relates]) for src_id, relates in message_dict['abs_idx_tree']])
                history_abs_idx_src.extend([src_id + history_abs_idx + 1 for src_id in message_dict['abs_idx_src']])
                history_abs_idx += (message_dict['abs_idx'] + 1)
                history_pos_idx += (message_dict['pos_idx'] + 1)

    def dbpedia_edge_list(self):
        edge_list = []
        for dbpediaId in self.dbpedia_subkg:
            for (related, relation) in self.dbpedia_subkg[dbpediaId]:
                edge_list.append((int(dbpediaId), int(related), relation))
                edge_list.append((int(related), int(dbpediaId), relation))
        return torch.tensor(edge_list, dtype=torch.long)

    def concept_edge_sets(self):
        edges = set()
        for line in self.concept_edges:
            lines = line.strip().split('\t')
            movie0 = self.concept2conceptIdx[lines[1].split('/')[0]]
            movie1 = self.concept2conceptIdx[lines[2].split('/')[0]]
            if lines[1].split('/')[0] in self.stopwords or lines[2].split('/')[0] in self.stopwords:
                continue
            edges.add((movie0, movie1))
            edges.add((movie1, movie0))
        edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return torch.tensor(edge_set, dtype=torch.long)

    def prepare_word2vec(self):
        """
        准备 Word2Vec 模型，对语料库进行处理并生成词向量 '<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3, '<split>':4, '<user>':5, '<movie>':6, '<mood>':7
        """
        corpus = []
        f = open(self.crs_data_path + '/' + 'train_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            for message in messages:
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                corpus.append(tokens)
        f = open(self.crs_data_path + '/' + 'test_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            for message in messages:
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                corpus.append(tokens)
        f = open(self.crs_data_path + '/' + 'valid_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            messages = lines['messages']
            for message in messages:
                tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                corpus.append(tokens)
        model = gensim.models.word2vec.Word2Vec(corpus, vector_size=300, min_count=1)
        word2wordIdx = {word: i + len(self.special_wordIdx) for i, word in enumerate(model.wv.index_to_key)}
        wordIdx2word = {i + len(self.special_wordIdx): word for i, word in enumerate(model.wv.index_to_key)}
        word2embedding = [[0] * 300] * len(self.special_wordIdx) + [model.wv[word] for word in word2wordIdx]
        json.dump(word2wordIdx, open(self.crs_data_path + '/redial_word2wordIdx.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(wordIdx2word, open(self.crs_data_path + '/redial_wordIdx2word.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)
        np.save(self.crs_data_path + '/redial_word2wordEmb.npy', word2embedding)

    def prepare_dbpedia_subkg(self):
        subkg = defaultdict(list)
        relation_cnt = defaultdict(int)
        relation_idx = {}
        datas = pkl.load(open(self.crs_data_path + '/dbpedia_db_db_subkg.pkl', 'rb'))
        for dbpediaId in datas:
            for (relation, related) in datas[dbpediaId]:
                relation_cnt[relation] += 1
        for relation in relation_cnt:
            if relation_cnt[relation] > 250:
                relation_idx[relation] = len(relation_idx)
        for dbpediaId in datas:
            for (relation, related) in datas[dbpediaId]:
                if relation in relation_idx:
                    assert dbpediaId < self.n_dbpedia + 1075
                    assert related < self.n_dbpedia + 1075
                    subkg[dbpediaId].append([related, relation_idx[relation]+18])

        f = open(self.crs_data_path + '/train_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            respondentWorkerId = lines['respondentWorkerId']
            initiatorWorkerId = lines['initiatorWorkerId']
            respondentQuestions = lines['respondentQuestions']
            initiatorQuestions = lines['initiatorQuestions']
            for movieId in respondentQuestions:
                rating = respondentQuestions[movieId]
                if self.movieId2movie[movieId] is not None:
                    dbpediaId = self.movie2dbpediaId[self.movieId2movie[movieId]]
                    assert self.userId2userIdx[str(initiatorWorkerId)]+self.n_dbpedia < self.n_dbpedia + 1075
                    assert dbpediaId < self.n_dbpedia + 1075
                    subkg[self.userId2userIdx[str(respondentWorkerId)]+self.n_dbpedia].append([dbpediaId, rating["suggested"] * 9 + rating["seen"] * 3 + rating["liked"]])
            for movieId in initiatorQuestions:
                rating = initiatorQuestions[movieId]
                if self.movieId2movie[movieId] is not None:
                    dbpediaId = self.movie2dbpediaId[self.movieId2movie[movieId]]
                    assert self.userId2userIdx[str(initiatorWorkerId)]+self.n_dbpedia < self.n_dbpedia + 1075
                    assert dbpediaId < self.n_dbpedia + 1075
                    subkg[self.userId2userIdx[str(initiatorWorkerId)]+self.n_dbpedia].append([dbpediaId, rating["suggested"] * 9 + rating["seen"] * 3 + rating["liked"]])

        json.dump(subkg, open(self.crs_data_path + '/dbpedia_subkg.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)

    def __getitem__(self, index):
        userIdx, dbpediaId, context_vector, context_mask, context_pos, context_vm, concept_mentioned, dbpedia_mentioned, related_mentioned, relation_mentioned, user_mentioned, response_vector, response_mask, response_pos, response_vm = self.datapre[index]
        concept_vector = np.zeros(self.n_concept)
        for con in concept_mentioned:
            if con != 0:
                concept_vector[con] = 1
        dbpedia_vector = np.zeros(self.n_dbpedia)
        for dbpedia in dbpedia_mentioned:
            if dbpedia != 0:
                dbpedia_vector[dbpedia] = 1
        return userIdx, torch.tensor(dbpediaId, dtype=torch.long), torch.tensor(context_vector, dtype=torch.long), context_mask, context_pos, torch.tensor(context_vm, dtype=torch.long), torch.tensor(concept_mentioned, dtype=torch.long), torch.tensor(dbpedia_mentioned, dtype=torch.long), torch.tensor(related_mentioned, dtype=torch.long), torch.tensor(relation_mentioned, dtype=torch.long), torch.tensor(user_mentioned, dtype=torch.long), torch.tensor(response_vector, dtype=torch.long), response_mask, response_pos, torch.tensor(response_vm, dtype=torch.long), concept_vector, dbpedia_vector

    def __len__(self):
        return len(self.datapre)

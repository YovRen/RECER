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
        self.n_user = args.n_user
        self.vocab_size = args.vocab_size
        self.n_relations = args.n_relations
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
                dbpedia_mentioned = []
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
                        relates_pos_idx.extend([pos_idx + 1, pos_idx + 2])
                        relates_abs_idx.extend([abs_idx + 1, abs_idx + 2])
                        abs_idx += 2
                    pos_idx_tree.append((pos_idx, relates_pos_idx))
                    abs_idx_tree.append((last_abs_idx, relates_abs_idx))
                    abs_idx_src.append(last_abs_idx)
                # 根据树构建其他内容input_ids&attention_mask
                for i, (token, relates) in enumerate(sent_tree):
                    if token in self.concept2conceptIdx.keys():
                        input_ids.append(self.vocab_size + self.concept2conceptIdx[token])
                        attention_mask.append(self.special_wordIdx['<concept>'])
                    elif token.isdigit() and (token in movieMentions.keys() and self.movieId2movie[token] in self.movie2dbpediaId.keys() or token in self.text2movie[message['text']]):
                        input_ids.append(self.vocab_size + self.n_concept + self.movie2dbpediaId[self.movieId2movie[token]])
                        attention_mask.append(self.special_wordIdx['<dbpedia>'])
                        dbpedia_mentioned.append(self.movie2dbpediaId[self.movieId2movie[token]])
                    else:
                        input_ids.append(self.word2wordIdx.get(token, self.special_wordIdx['<unk>']))
                        attention_mask.append(self.special_wordIdx['<word>'])
                    for j, relate in enumerate(relates):
                        input_ids.append(self.vocab_size + self.n_concept + relate[0])
                        attention_mask.append(self.special_wordIdx['<related>'])
                        input_ids.append(self.vocab_size + self.n_concept + self.n_dbpedia + relate[1])
                        attention_mask.append(self.special_wordIdx['<relation>'])
                # 合并每条消息
                userIdx = int(self.userId2userIdx[str(message['senderWorkerId'])])
                if num == 0:
                    message_dict = {'userIdx': userIdx, 'input_ids': input_ids, 'attention_mask': attention_mask, 'dbpedia_mentioned': dbpedia_mentioned, 'abs_idx_tree': abs_idx_tree, 'pos_idx_tree': pos_idx_tree, 'abs_idx_src': abs_idx_src, 'pos_idx': pos_idx, 'abs_idx': abs_idx}
                    message_list.append(message_dict)
                elif userIdx == message_list[-1]['userIdx']:
                    message_list[-1]['dbpedia_mentioned'] += dbpedia_mentioned
                    message_list[-1]['input_ids'] += input_ids
                    message_list[-1]['attention_mask'] += attention_mask
                    message_list[-1]['pos_idx_tree'] += [(src_id + message_list[-1]['pos_idx'] + 1, [relate + message_list[-1]['pos_idx'] + 1 for relate in relates]) for src_id, relates in pos_idx_tree]
                    message_list[-1]['abs_idx_tree'] += [(src_id + message_list[-1]['abs_idx'] + 1, [relate + message_list[-1]['abs_idx'] + 1 for relate in relates]) for src_id, relates in abs_idx_tree]
                    message_list[-1]['abs_idx_src'] += [src_id + message_list[-1]['abs_idx'] + 1 for src_id in abs_idx_src]
                    message_list[-1]['pos_idx'] += (pos_idx + 1)
                    message_list[-1]['abs_idx'] += (abs_idx + 1)
                else:
                    message_list[-1]['input_ids'] = [userIdx + self.vocab_size + self.n_concept + self.n_dbpedia + self.n_relations] + message_list[-1]['input_ids'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['attention_mask'] = [self.special_wordIdx['<user>']] + message_list[-1]['attention_mask'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['pos_idx_tree'] = [(0, [])] + [(src_id + 1, [relate + 1 for relate in relates]) for src_id, relates in message_list[-1]['pos_idx_tree']] + [(message_list[-1]['pos_idx'] + 2, [])]
                    message_list[-1]['abs_idx_tree'] = [(0, [])] + [(src_id + 1, [relate + 1 for relate in relates]) for src_id, relates in message_list[-1]['abs_idx_tree']] + [(message_list[-1]['abs_idx'] + 2, [])]
                    message_list[-1]['abs_idx_src'] = [0] + [src_id + 1 for src_id in message_list[-1]['abs_idx_src']] + [message_list[-1]['abs_idx'] + 2]
                    message_list[-1]['pos_idx'] += 2
                    message_list[-1]['abs_idx'] += 2
                    message_dict = {'userIdx': userIdx, 'input_ids': input_ids, 'attention_mask': attention_mask, 'dbpedia_mentioned': dbpedia_mentioned, 'abs_idx_tree': abs_idx_tree, 'pos_idx_tree': pos_idx_tree, 'abs_idx_src': abs_idx_src, 'pos_idx': pos_idx, 'abs_idx': abs_idx}
                    message_list.append(message_dict)
                if num == len(messages) - 1:
                    message_list[-1]['input_ids'] = [userIdx + self.vocab_size + self.n_concept + self.n_dbpedia + self.n_relations] + message_list[-1]['input_ids'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['attention_mask'] = [self.special_wordIdx['<user>']] + message_list[-1]['attention_mask'] + [self.special_wordIdx['<mood>']]
                    message_list[-1]['pos_idx_tree'] = [(0, [])] + [(src_id + 1, [relate + 1 for relate in relates]) for src_id, relates in message_list[-1]['pos_idx_tree']] + [(message_list[-1]['pos_idx'] + 2, [])]
                    message_list[-1]['abs_idx_tree'] = [(0, [])] + [(src_id + 1, [relate + 1 for relate in relates]) for src_id, relates in message_list[-1]['abs_idx_tree']] + [(message_list[-1]['abs_idx'] + 2, [])]
                    message_list[-1]['abs_idx_src'] = [0] + [src_id + 1 for src_id in message_list[-1]['abs_idx_src']] + [message_list[-1]['abs_idx'] + 2]
                    message_list[-1]['pos_idx'] += 2
                    message_list[-1]['abs_idx'] += 2

            # 将message_list中所有的消息整合起来
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
                    response_attention_mask = np.array(message_dict['attention_mask'])[np.isin(message_dict['attention_mask'], [1, 4, 5])]
                    response_input_ids = np.array(message_dict['input_ids'])[np.isin(message_dict['attention_mask'], [1, 4, 5])]
                    response_input_ids[response_input_ids >= self.vocab_size + self.n_concept] = self.special_wordIdx['<dbpedia>']
                    assert history_abs_idx == history_abs_idx_src[-1]
                    if len(message_dict['dbpedia_mentioned']) == 0:
                        assert history_abs_idx == history_abs_idx_src[-1]
                        self.datapre.append([userIdx, 0, history_input_ids.copy(), history_attention_mask.copy(), response_input_ids.tolist(), response_attention_mask.tolist(), history_pos_idx_tree.copy(), history_pos_idx, history_abs_idx_tree.copy(), history_abs_idx, history_abs_idx_src.copy()])
                    for dbpediaId in message_dict['dbpedia_mentioned']:
                        assert history_abs_idx == history_abs_idx_src[-1]
                        self.datapre.append([userIdx, dbpediaId, history_input_ids.copy(), history_attention_mask.copy(), response_input_ids.tolist(), response_attention_mask.tolist(), history_pos_idx_tree.copy(), history_pos_idx, history_abs_idx_tree.copy(), history_abs_idx, history_abs_idx_src.copy()])
                if len(history_attention_mask) != 0:
                    history_input_ids.extend([self.special_wordIdx['<split>']])
                    history_attention_mask.extend([self.special_wordIdx['<split>']])
                    history_pos_idx_tree.extend([(history_pos_idx + 1, [])])
                    history_abs_idx_tree.extend([(history_abs_idx + 1, [])])
                    history_abs_idx_src.extend([history_abs_idx + 1])
                    history_abs_idx += 1
                    history_pos_idx += 1
                history_input_ids.extend(message_dict['input_ids'])
                history_attention_mask.extend(message_dict['attention_mask'])
                history_pos_idx_tree.extend([(src_id + history_pos_idx + 1, [relate + history_pos_idx + 1 for relate in relates]) for src_id, relates in message_dict['pos_idx_tree']])
                history_abs_idx_tree.extend([(src_id + history_abs_idx + 1, [relate + history_abs_idx + 1 for relate in relates]) for src_id, relates in message_dict['abs_idx_tree']])
                history_abs_idx_src.extend([src_id + history_abs_idx + 1 for src_id in message_dict['abs_idx_src']])
                history_abs_idx += (message_dict['abs_idx'] + 1)
                history_pos_idx += (message_dict['pos_idx'] + 1)
        # self.datapre = self.datapre[:len(self.datapre) // 5]

    def dbpedia_edge_list(self):
        edge_list = []
        for dbpediaId in self.dbpedia_subkg:
            for (related, relation) in self.dbpedia_subkg[dbpediaId]:
                edge_list.append((int(dbpediaId), int(related), relation))
                edge_list.append((int(related), int(dbpediaId), relation))
        return torch.tensor(edge_list, dtype=torch.long)

    def concept_edge_sets(self):
        nodes = set()
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
        for mode in ['train', 'test', 'valid']:
            f = open(self.crs_data_path + '/' + mode + '_data.jsonl', encoding='utf-8')
            for case in tqdm(f):
                lines = json.loads(case.strip())
                messages = lines['messages']
                for message in messages:
                    tokens = nltk.word_tokenize(message['text'].replace('@', ''))
                    new_tokens = []
                    for token in tokens:
                        if token not in self.concept2conceptIdx.keys() and not token.isdigit():
                            new_tokens.append(token)
                    corpus.append(new_tokens)
        model = gensim.models.word2vec.Word2Vec(corpus, vector_size=300, min_count=5)
        word2wordIdx = {word: i + len(self.special_wordIdx) for i, word in enumerate(model.wv.index_to_key)}
        word2embedding = [[0] * 300] * len(self.special_wordIdx) + [model.wv[word] for word in word2wordIdx]
        json.dump(word2wordIdx, open(self.crs_data_path + '/redial_word2wordIdx.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)
        np.save(self.crs_data_path + '/redial_word2wordEmb.npy', word2embedding)

    def prepare_concept2conceptIdx(self):
        concept2conceptIdx = {}
        for line in self.concept_edges:
            lines = line.strip().split('\t')
            node1 = lines[1].split('/')[0].lower()
            node2 = lines[2].split('/')[0].lower()
            print(len(concept2conceptIdx))
            if node1 not in concept2conceptIdx:
                concept2conceptIdx[node1] = len(concept2conceptIdx)
            if node2 not in concept2conceptIdx:
                concept2conceptIdx[node2] = len(concept2conceptIdx)
        json.dump(concept2conceptIdx, open(self.crs_data_path + '/concept_concept2conceptIdx.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)

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
                    subkg[dbpediaId].append([related, relation_idx[relation] + 18])

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
                    assert self.userId2userIdx[str(initiatorWorkerId)] + self.n_dbpedia < self.n_dbpedia + 1075
                    assert dbpediaId < self.n_dbpedia + 1075
                    subkg[self.userId2userIdx[str(respondentWorkerId)] + self.n_dbpedia].append([dbpediaId, rating["suggested"] * 9 + rating["seen"] * 3 + rating["liked"]])
            for movieId in initiatorQuestions:
                rating = initiatorQuestions[movieId]
                if self.movieId2movie[movieId] is not None:
                    dbpediaId = self.movie2dbpediaId[self.movieId2movie[movieId]]
                    assert self.userId2userIdx[str(initiatorWorkerId)] + self.n_dbpedia < self.n_dbpedia + 1075
                    assert dbpediaId < self.n_dbpedia + 1075
                    subkg[self.userId2userIdx[str(initiatorWorkerId)] + self.n_dbpedia].append([dbpediaId, rating["suggested"] * 9 + rating["seen"] * 3 + rating["liked"]])

        json.dump(subkg, open(self.crs_data_path + '/dbpedia_subkg.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)

    def __getitem__(self, index):
        userIdx, dbpediaId, history_input_ids, history_attention_mask, response_input_ids, response_attention_mask, history_pos_idx_tree, history_pos_idx, history_abs_idx_tree, history_abs_idx, history_abs_idx_src = self.datapre[index]
        assert history_abs_idx == history_abs_idx_src[-1]
        context_vector = np.array((history_input_ids + [self.special_wordIdx['<eos>']] + [self.special_wordIdx['<pad>']] * (self.max_c_length - len(history_input_ids)))[:self.max_c_length])
        context_mask = np.array((history_attention_mask + [self.special_wordIdx['<eos>']] + [self.special_wordIdx['<pad>']] * (self.max_c_length - len(history_attention_mask)))[:self.max_c_length])
        response_vector = np.array((response_input_ids + [self.special_wordIdx['<pad>']] * (self.max_r_length - len(response_input_ids)))[:self.max_r_length])
        # Calculate visible matrix
        context_pos = []
        for src_id, relates in history_pos_idx_tree:
            context_pos.append(src_id)
            for relate in relates:
                context_pos.append(relate)
        context_pos = np.array((context_pos + [self.max_c_length - 1] * (self.max_c_length - len(context_pos)))[:self.max_c_length])
        # 构建可视矩阵
        context_vm = np.zeros((history_abs_idx + 1, history_abs_idx + 1))
        history_dbpedia_count = []
        for src_id, relates in history_abs_idx_tree:
            context_vm[src_id, history_abs_idx_src + relates] = 1
            for i in range(0, len(relates), 2):
                for j, (epoch_src_id, epoch_relates) in enumerate(history_dbpedia_count):
                    context_vm[relates[i], [epoch_src_id] + epoch_relates] = 0.9 ** (len(history_dbpedia_count) - j)
                    context_vm[relates[i + 1], [epoch_src_id] + epoch_relates] = 0.9 ** (len(history_dbpedia_count) - j)
                context_vm[relates[i], [src_id] + relates[i:i + 2]] = 1
                context_vm[relates[i + 1], [src_id] + relates[i:i + 2]] = 1
            if len(relates) > 0:
                history_dbpedia_count.append((src_id, relates))
        if history_abs_idx < self.max_c_length:
            context_vm = np.pad(context_vm, ((0, self.max_c_length - history_abs_idx - 1), (0, self.max_c_length - history_abs_idx - 1)), 'constant')
        else:
            context_vm = context_vm[:self.max_c_length, :self.max_c_length]

        concept_vector = np.zeros(self.n_concept)
        concept_vector[context_vector[context_mask == self.special_wordIdx['<concept>']] - self.vocab_size] = 1
        dbpedia_vector = np.zeros(self.n_dbpedia)
        dbpedia_vector[context_vector[context_mask == self.special_wordIdx['<dbpedia>']] - self.vocab_size-self.n_concept] = 1
        user_vector = np.zeros(self.n_user)
        user_vector[context_vector[context_mask == self.special_wordIdx['<user>']] - self.vocab_size-self.n_concept-self.n_dbpedia-self.n_relations] = 1
        return userIdx, torch.tensor(dbpediaId, dtype=torch.long), torch.tensor(context_vector, dtype=torch.long), torch.tensor(context_mask, dtype=torch.long), context_pos, torch.tensor(context_vm, dtype=torch.float64), torch.tensor(response_vector, dtype=torch.long), concept_vector, dbpedia_vector, user_vector

    def __len__(self):
        return len(self.datapre)

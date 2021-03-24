"""
    @author: Jevis_Hoo
    @Date: 2021/2/20
    @Description: 
"""
from tqdm import tqdm
import numpy as np
import pickle
import os
import random
import jieba.posseg as pseg
from snippts import split_text, find_lcsubstr
import re
import codecs
import pandas as pd
from config import Config
from gensim.summarization import bm25

config = Config()
max_len = 360

stop_flag = ['x', 'c', 'u']
stop_words = 'stop_words.txt'
stopwords = codecs.open(stop_words, 'r', encoding='utf-8').readlines()
stopwords = [w.strip() for w in stopwords]


def load_context(file_path):
    doc_id2context = {}
    f = True
    for line in open(file_path):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        doc_id2context[r[0]] = '\t'.join([r[i] for i in range(1, len(r))])

    return doc_id2context


def get_para_df(doc_id2context):
    deal_text_list = []
    for doc_id in tqdm(doc_id2context):
        text = doc_id2context[doc_id]
        body = text
        body = body.strip()
        if body.__len__() <= max_len:
            deal_text = [body]
        elif re.search('\xa0|\u3008|\u2002|\u2003|\uFF01|\uFF1F|\uFF61|\u3002', body):
            "段落有明显分割符"
            deal_text = split_text(body, max_len, greedy=False)[0]
        else:
            "使用空格作为分割符"
            split_pat = '([{ }]”?)'
            deal_text = split_text(body, max_len, split_pat=split_pat, greedy=False)[0]

        i = 0
        for text in deal_text:
            res_dict = {'para_id': doc_id + '_' + str(i), 'doc_id': doc_id, 'text': text}
            deal_text_list.append(res_dict)
            i += 1
        para_df = pd.DataFrame(deal_text_list)
        return para_df


def refind_answer(train_df, para_df):
    answer_list = []
    for i in tqdm(range(len(train_df))):
        doc_id = train_df['doc_id'][i]
        answer = train_df['answer'][i].strip()

        try:
            contexts = para_df[para_df['doc_id'] == doc_id]['text']
        except:
            contexts = []

        flag = 0
        for context in contexts:
            if context.count(answer):
                flag = 1
        if flag == 1:
            answer_list.append(answer)
        else:
            _answer = ''
            for context in contexts:
                lcs = find_lcsubstr(context, answer)[0]
                if len(lcs) >= _answer.__len__():
                    _answer = lcs
            answer_list.append(answer)

    return answer_list


def tokenization(text):
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


def load_corpus(context_df):
    corpus = []
    para_id_list = []

    for i in tqdm(range(len(context_df))):
        text_list = tokenization(context_df['text'][i])
        corpus.append(text_list.copy())

        para_id_list.append(context_df['para_id'][i])
    return corpus, para_id_list


def generate_para(train_df, para_df, id2para, para_id_list, bm25_model, dev_num=0):
    doc_train = []

    for i in tqdm(range(len(train_df))):
        answer = train_df['answer'].iloc[i]
        qid = train_df['id'].iloc[i]
        question = train_df['question'].iloc[i]
        doc_id = train_df['doc_id'].iloc[i]
        context = para_df[para_df['doc_id'] == doc_id]['text']

        """Pos"""
        pos_text = []
        for text in context:
            if text.count(answer):
                start = text.find(answer)
                doc_dict = {'q_id': qid, 'query': question, 'start': start,
                            'context': context, 'answer': answer, 'score': 1}
                pos_text.append(text)
                doc_train.append(doc_dict)
        """Neg"""
        query = tokenization(question)
        scores = bm25_model.get_scores(query)
        scores = np.array(scores)
        sort_index = np.argsort(-scores)[:50]
        para_ids = [para_id_list[i] for i in sort_index]

        if dev_num:
            para_ids = random.sample(para_ids, dev_num)
        else:
            para_ids = random.sample(para_ids, len(pos_text))

        for idx in para_ids:
            neg_text = id2para[idx]
            while neg_text in pos_text:
                neg_text = id2para[random.sample(para_ids, 1)[0]]
            doc_dict = {'q_id': qid, 'query': question, 'start': -1,
                        'context': neg_text, 'answer': answer, 'score': 0}
            doc_train.append(doc_dict)
    doc_df = pd.DataFrame(doc_train)
    return doc_df


def test_generate_para(test_df, id2para, para_id_list, bm25_model, dev_num=25):
    doc_train = []

    for i in tqdm(range(len(test_df))):
        qid = test_df['id'].iloc[i]
        question = test_df['question'].iloc[i]

        query = tokenization(question)
        scores = bm25_model.get_scores(query)
        scores = np.array(scores)
        sort_index = np.argsort(-scores)[:50]
        para_ids = [para_id_list[i] for i in sort_index]
        para_id = random.sample(para_ids, dev_num)

        for idx in para_id:
            neg_text = id2para[idx]
            doc_dict = {'q_id': qid, 'query': question, 'start': -1,
                        'context': neg_text, 'answer': '', 'score': 0}
            doc_train.append(doc_dict)

    doc_df = pd.DataFrame(doc_train)
    return doc_df


def main():
    doc_id2context = load_context('train.json')
    para_df = get_para_df(doc_id2context)

    corpus, para_id_list = load_corpus(para_df)

    bm25_model = bm25.BM25(corpus)

    train_df = []
    dev_df = []
    id2para = dict(zip(list(para_df['para_id']), list(para_df['text'])))
    para_train = generate_para(train_df, para_df, id2para, para_id_list, bm25_model)
    para_dev = generate_para(dev_df, para_df, id2para, para_id_list, bm25_model, dev_num=25)
    para_train['score'] = 0
    para_dev['score'] = 0

    para_train.to_csv('train.csv', index=None, encoding='utf-8')
    para_dev.to_csv('dev.csv', index=None, encoding='utf-8')

    test_df = []
    test_df = test_generate_para(test_df, id2para, para_id_list, bm25_model, dev_num=25)
    test_df.to_csv('test.csv', index=None, encoding='utf-8')


if __name__ == "__main__":
    main()

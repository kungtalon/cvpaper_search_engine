import os
import re
import json
import nltk
import numpy as np
import pandas as pd
import pickle as pkl
import pyterrier as pt
from os.path import join as pjoin
from collections import Counter


FIELDS = ['title', 'abstract', 'subsections', 'authors']
WEIGHTS = {
    'title': 0.5,
    'abstract': 0.4,
    'subsections': 0.1,
    'authors': 1
}
SAMPLE_INDEXES = [i for start, end in [[0,50]] for i in range(start, end)]

def load_df_pickle(path):
    with open(path, 'rb') as f:
        df = pkl.load(f)
    print('dataframe loaded, columns: ', df.columns)
    return df

def get_index_data(input_df, output_path):
    indexes = {}

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for field in FIELDS:
        field_path = pjoin(output_path, field)
        if not os.path.exists(pjoin(field_path, "data.properties")):
            indexer = pt.DFIndexer(field_path, overwrite=True)
            index_ref = indexer.index(input_df[field], input_df['docno'])
        else:
            index_ref = pjoin(field_path, "data.properties")
        indexes[field] = pt.IndexFactory.of(index_ref)
    return indexes

def retrieve(query, indexes, weights):
    results = []
    for field in FIELDS:
        br = pt.BatchRetrieve(indexes[field], wmodel="BM25")
        result = br.search(query)[['docno', 'score']]
        result['score'] *= weights[field]
        results.append(result)
    return pd.concat(results).groupby('docno').sum('score').sort_values(by='score', ascending=False).reset_index()

def min_edit_dist(s, t):
    l1 = len(s)
    l2 = len(t)
    dp = [[100000]*(l2+1) for _ in range(l1+1)]
    dp[0][0] = 0
    for i in range(l1+1):
        dp[i][0] = i
    for j in range(l2+1):
        dp[0][j] = j
    for i in range(l1):
        for j in range(l2):
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j]
            else:
                dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
    return dp[-1][-1]

def print_highlight_info(info, query):
    q_words = query.strip().split()
    ps = nltk.stem.PorterStemmer()
    for line in info:
        line_highlited = []
        terms = line.split(' ')
        for term in terms:
            flag = False
            for q in q_words:
                if ps.stem(re.sub('[,\.\?()]', '', term).lower()) == ps.stem(q) or min_edit_dist(re.sub('[,\.\?()]', '', term), q) <= 1:
                    flag = True
                    break
            if flag:
                term = '\033[31;1m' + term + '\033[0m'
            line_highlited.append(term)
        print(' '.join(line_highlited))

def annotate(data, queries, indexes):
    for query in queries:
        fname = pjoin('./annotated', query + '.json')
        if os.path.exists(fname):
            continue
        docs = retrieve(query, indexes, WEIGHTS)
        candidates = docs.iloc[SAMPLE_INDEXES]
        candidates_detail = candidates.merge(data, on='docno')
        json_list = []
        labels = []
        for i, line in candidates_detail.iterrows():
            print('*****************************************')
            print('[NO]: ' + str(i))
            print('[QUERY]: ' + query)
            print_highlight_info(['[' + field.upper() + ']: ' + line[field] for field in FIELDS], query)
            # info = '\n'.join(['[QUERY]: ' + query] + ['[' + field.upper() + ']: ' + line[field] for field in FIELDS])
            # print(info)
            while True:
                inp = input('Please rate the relevance of the document: (1 - 5) ')
                if not inp.isnumeric():
                    continue
                rating = int(inp)
                if rating in [1,2,3,4,5]:
                    break
            json_list.append({
                'query': query,
                'docno': line['docno'],
                'label': rating
            })
            labels.append(rating)
        print(f'The count of labels for query "{query}"')
        print(dict(Counter(labels)))
        print('====================Finished====================')
        with open(fname, 'w') as f:
            json.dump(json_list, f)

def get_queries(query_file):
    with open(query_file, 'r') as f:
        txt = f.read().split('\n')
    while True:
        name = input('input your name:')
        if name == 'jzl':
            print('jzl is annotating!')
            return txt[1::2]
        elif name == 'yy':
            print('yy is annotating!')
            return txt[::2]

def main():
    data = load_df_pickle('./cleandatanew.pkl')
    if 'docno' in data.columns:
        data['docno'] = data['docno'].astype('str')
    else:
        data = data.reset_index()
        data['docno'] = data['index'].astype('str')
    data['subsections'] = data['subsections'].apply(lambda x: x.replace(';', ' '))
    if not pt.started():
        pt.init(version=5.6, helper_version='0.0.6')
    indexes = get_index_data(data, './index')

    os.makedirs('./annotated', exist_ok=True)
    queries = get_queries('./queries.txt')
    annotate(data, queries, indexes)

if __name__ == '__main__':
    main()

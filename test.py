import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from indexer import PaperRetrieval, EmptyRetrievalError

ir = PaperRetrieval(index_root='./index', wv_path='./word2vec.wv',
                        embedding_path='./word2vec_embedding_df.csv', json_path='./training_json/',
                        model_path='./gbm_save.pkl')


def get_retrieval_iterator(results):
    info = []
    for i, data in enumerate(results.values):
        _, title, authors, abstract = data
        info.append('*****************************************')
        info.append('[NO]: ' + str(i+1))
        info.append('[TITLE]: ' + title)
        info.append('[AUTHOR]: ' + authors)
        info.append('[ABSTRACT]: ' + abstract)
        if i % 3 == 2:
            yield info
            info = []
    yield info


if len(sys.argv) == 2 and sys.argv[1] == '-i':
    while True:
        inp = input('Please input the query or [q for quit]:')
        if inp == 'q':
            exit(0)
        try:
            results = ir.search(inp)
        except EmptyRetrievalError:
            print('Sorry! No results found for this query!')
            continue
        return_info = get_retrieval_iterator(results)
        for infos in return_info:
            print('\n'.join(infos))
            tmp_inp = ''
            while tmp_inp not in ['c', 'r']:
                tmp_inp = input('Type [c for continue browsing, r for search for a new query]:')
            if tmp_inp == 'r':
                break


test_json_path = './annotated/'
test_file_names = os.listdir(test_json_path)
ndcg_recall = 0
ndcg_rank = 0
cnt = 0
for fname in test_file_names:
    json_df = pd.read_json(test_json_path + fname)
    json_df['docno'] = json_df['docno'].astype('str')
    dic = {k: v for k, v in json_df[['docno', 'label']].values}
    preds = ir.search(json_df['query'][0])
    import pdb; pdb.set_trace()
    y_true = preds.apply(lambda x: dic[x['docno']], axis=1).values.reshape(1, -1)
    y_score = 1 / np.arange(1, 51).reshape(1, -1)
    ndcg_rank += ndcg_score(y_true, y_score)

    nol2r_preds = ir.search(json_df['query'][0], {'no_l2r':True})
    y_true_nol2r = nol2r_preds.apply(lambda x: dic.get(x['docno'], -1), axis=1).values
    y_true_nol2r = y_true_nol2r[y_true_nol2r != -1].reshape(1, -1)
    ndcg_recall += ndcg_score(y_true_nol2r, y_score)
    cnt += 1

print('test ndcg: %s, test ndcg for bm25 : %s' % (ndcg_rank / cnt, ndcg_recall / cnt))

features = [
    'dot_t', 'dot_a', 'dot_s',
    'cos_t', 'cos_a', 'cos_s',
    'norm_t', 'norm_a', 'norm_s',
    'tfidf_t', 'tfidf_a', 'tfidf_s',
    'bm25_t', 'bm25_a', 'bm25_s',
    'cm_t', 'cm_a', 'cm_s',
    'year', 'conference', 'workshop', 'supp', 'authors'
]

values = ir.model.booster_.feature_importance()
fi = sorted(zip(values, features), reverse=True)
y, x = zip(*fi)

plt.bar(x, y)
plt.xticks(rotation=90)
plt.show()
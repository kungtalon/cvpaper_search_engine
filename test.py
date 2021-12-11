import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from indexer import PaperRetrieval



ir = PaperRetrieval(index_root='./index', wv_path='./word2vec.wv',
                        embedding_path='./word2vec_embedding_df.csv', json_path='./training_json/',
                        model_path='./gbm_save.pkl')

test_json_path = './annotated/'
test_file_names = os.listdir(test_json_path)
ndcg = 0
cnt = 0
for fname in test_file_names:
    json_df = pd.read_json(test_json_path + fname)
    json_df['docno'] = json_df['docno'].astype('str')
    dic = {k: v for k, v in json_df[['docno', 'label']].values}
    preds = ir.search(json_df['query'][0])
    # import pdb; pdb.set_trace()
    y_true = preds.apply(lambda x: dic[x['docno']], axis=1).values.reshape(1, -1)
    y_score = 1 / np.arange(1, 51).reshape(1, -1)
    ndcg += ndcg_score(y_true, y_score)
    cnt += 1

print('test ndcg: %s' % (ndcg / cnt))



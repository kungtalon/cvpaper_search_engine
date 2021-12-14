import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import ndcg_score

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

def test_iter():
    from indexer import PaperRetrieval, EmptyRetrievalError
    ir = PaperRetrieval(index_root='./index', wv_path='./word2vec.wv',
                        embedding_path='./word2vec_embedding_df.csv', json_path='./training_json/',
                        model_path='./gbm_save.lgb')
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

if len(sys.argv) == 2 and sys.argv[1] == '-i':
    test_iter()

def test(ir):
    test_json_path = './annotated/'
    test_file_names = os.listdir(test_json_path)
    ndcg_recall = [0, 0, 0]
    ndcg_rank = [0, 0, 0]
    cnt = 0
    ys = []
    ys_nol2r = []
    for fname in test_file_names:
        json_df = pd.read_json(test_json_path + fname)
        json_df['docno'] = json_df['docno'].astype('str')
        dic = {k: v for k, v in json_df[['docno', 'label']].values}
        preds = ir.search(json_df['query'][0])
        # import pdb; pdb.set_trace()
        y_true = preds.apply(lambda x: dic.get(x['docno'], -1), axis=1).values
        ys.append(y_true[y_true != -1].reshape(1, -1))

        nol2r_preds = ir.search(json_df['query'][0], {'no_l2r':True})
        y_true_nol2r = nol2r_preds.apply(lambda x: dic.get(x['docno'], -1), axis=1).values
        ys_nol2r.append(y_true_nol2r[y_true_nol2r != -1].reshape(1, -1))

    y_score = 1 / np.arange(1, ys[0].shape[1]+1).reshape(1, -1).repeat(len(test_file_names), axis=0)
    k = [5, 10, 20]
    ys = np.concatenate(ys, axis=0)
    ys_nol2r = np.concatenate(ys_nol2r, axis=0)
    for i in range(3):
        ndcg_rank[i] = ndcg_score(ys, y_score, k=k[i])
        ndcg_recall[i] = ndcg_score(ys_nol2r, y_score, k=k[i])
    print('test ndcg: %s, test ndcg for bm25 : %s' % (ndcg_rank, ndcg_recall))

def plot_importance(model):
    features = [
        'dot_t', 'dot_a', 'dot_s',
        'cos_t', 'cos_a', 'cos_s',
        'norm_t', 'norm_a', 'norm_s',
        'tfidf_t', 'tfidf_a', 'tfidf_s',
        'bm25_t', 'bm25_a', 'bm25_s',
        'cm_t', 'cm_a', 'cm_s',
        'year', 'conference', 'workshop', 
        'supp', 'score', 'authors'
    ]
    
    values = model.feature_importance(importance_type='gain')
    fi = sorted(zip(values, features), reverse=True)
    y, x = zip(*fi)
    
    plt.bar(y[:-1], x[:-1])
    plt.xticks(rotation=90)
    plt.show()

def label_distribution(json_path):
    train_file_names = os.listdir(json_path)
    cnt = {i:0 for i in range(1,6)}
    for fname in train_file_names:
        json_df = pd.read_json(os.path.join(json_path, fname))
        counts = Counter(json_df['label'])
        cnt = {i : cnt[i] + counts[i] for i in range(1, 6)}
    total = sum(cnt.values())
    cnt = {i : cnt[i] / total for i in range(1, 6)}
    print(cnt)

def plot_model_results():
    #test ndcg:[0.8502954288976735, 0.8253090584034174, 0.8260557877102949]
    #test ndcg for bm25:[0.8006712169387947, 0.8101556628356219, 0.8241520806752731]
    #logistic: [0.559771010423048, 0.5842655132220035, 0.6832164470572264]
    #point lgbm 0.6627813239470598, 0.6769866245757301, 0.7289770024166153]
    import seaborn as sns; sns.set()
    columns = ['BM25', 'LGBM_list', 'LGBM_point', 'LR']
    results = [
        [0.8006712169387947, 0.8101556628356219, 0.8241520806752731],
        [0.8502954288976735, 0.8253090584034174, 0.8260557877102949],
        [0.6627813239470598, 0.6769866245757301, 0.7289770024166153],
        [0.559771010423048, 0.5842655132220035, 0.6832164470572264],
    ]
    results = np.array(results).T
    df = pd.DataFrame(results, columns=columns)
    splot = df.plot(kind='bar', color=['b', 'y', 'r', 'g'])
    for p in splot.patches:
        splot.annotate(format(p.get_height()*100, '2.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height() + 0.05), 
                   ha = 'center', va = 'center', 
                   size=12,
                   xytext = (0, -12), 
                   textcoords = 'offset points')
    plt.ylim(0, 1)
    plt.legend()
    plt.ylabel('Test nDCG(×100)')
    plt.xticks(range(3), labels=['ndcg_5', 'ndcg_10', 'ndcg_20'], rotation=0)
    plt.yticks([0.0, 0.2,0.4,0.6,0.8,1.0], labels=[0,20, 40,60,80,100])
    plt.show()

def plot_hyperparameters():
    # 250 : [0.8202619115568333, 0.7819168317130917, 0.7950530033853338]
    # 200 : [0.8565681343512465, 0.8298648140139626, 0.8481365467818849]
    # 150 : [0.797697503072131, 0.7832461535118415, 0.8015466065491825]
    # 100 : [0.8089773387754533, 0.7791320969315374, 0.8021327286364042]
    # all : [0.8502954288976735, 0.8253090584034174, 0.8260557877102949]
    import seaborn as sns; sns.set()
    # data = [
    #     [0.8202619115568333, 0.7819168317130917, 0.7950530033853338],
    #     [0.8565681343512465, 0.8298648140139626, 0.8481365467818849],
    #     [0.797697503072131, 0.7832461535118415, 0.8015466065491825],
    #     [0.8089773387754533, 0.7791320969315374, 0.8021327286364042],
    #     [0.8502954288976735, 0.8253090584034174, 0.8260557877102949]
    # ]
    data = [
        [0.798730588229321, 0.8020859449329996, 0.800993418948404],
        [0.7913517938300433, 0.7967713089614463, 0.8280307819582561],
        [0.8119997264140594, 0.7940471696594027, 0.8238827374387915],
        [0.8502954288976735, 0.8253090584034174, 0.8260557877102949],
        [0.8305000792988547, 0.7980043772130585, 0.8190607994902503],
    ]
    data = np.array(data)
    labels = ['5', '10', '20']
    plt.figure()
    for i in range(3):
        plt.plot(range(5), data[:, i], label='ndcg_'+labels[i])
    # plt.xticks(range(5), labels=['>250', '>200', '>150', '>100', 'ALL'])
    plt.xticks(range(5), labels=['32', '64', '128', '256', '384'])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_hyperparameters()
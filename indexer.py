import os
import re
import copy
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
import pyterrier as pt
import pickle as pkl
from functools import reduce
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from os.path import join as pjoin
from spelling_corrector import SpellingCorrector
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
import test


FIELDS = ['title', 'abstract', 'subsections', 'authors']

class EmptyRetrievalError(Exception):
    pass

class FeatureExtractor():
    def __init__(self, df, indexes, word2vec, user_args={}, is_training=False):
        self.indexes = indexes  #dictionary, key is the field_name, value is the corresponding index
        self.user_args = user_args
        self.data = df
        self.query_embeddings_dict = self._get_query_embedding(word2vec)
        self.is_training = is_training

    def _get_query_embedding(self, word2vec):
        ps = PorterStemmer()
        # stop_words = set(stopwords.words('english'))
        # no filtering of stop words for query
        def transform_query(query):
            hyphen_words = re.findall('\w+-\w+', query)
            hwords = reduce(lambda x, y: x + y, [[*word.split('-'), word.replace('-', '')] for word in hyphen_words], [])
            q_list = [term.lower() for term in query.split() + hwords]
            q_list = [ps.stem(term) for term in q_list]
            embedding = np.sum([word2vec[term] for term in q_list if term in word2vec.key_to_index], axis=0)
            if not isinstance(embedding, np.ndarray):
                embedding = np.zeros(word2vec.vector_size)
            return embedding

        q_dict = {qid: query for qid, query in self.data[['qid', 'query']].values}
        q_embed_dict = {qid: transform_query(query) for qid, query in q_dict.items()}
        return q_embed_dict

    def get_features(self):
        # missing values will be filled with np.nan
        # embeddings title, abstract, method 16d
        # title, abstract, subsection, method bm25 tf-idf CoordinateMatch
        # publish time, conference, is_workshop, has_supp, 
        # author hit-rate, try bi-gram
        feature_extractor_funcs = {
            # 'feature_name': self.func,
            'embedding_dists': self._cal_embedding_dists,
            'pyterrier_ranking': self._pyterrier_rank,
            'doc_property': self._get_doc_property,
            'author_name_match': self._match_author_name
        }

        features = pd.DataFrame()
        for feature_name, feature_transform in feature_extractor_funcs.items():
            features[feature_name] = feature_transform()
        
        self.data['features'] = features.apply(lambda x: np.concatenate(x.values), axis=1)

        if 'feat_selection' in self.user_args and self.user_args['feat_selection'] > 0:
            # the indexes of features ordered by the feature importance in lgbm
            idx = [22, 2, 12, 1, 13, 0, 5, 7, 9, 4, 3, 10, 6, 15, 8, 14, 18, 16, 11, 19, 21, 20, 17, 23]
            if self.user_args['feat_selection'] == 250:
                select_idx = idx[:5]
            elif self.user_args['feat_selection'] == 200:
                select_idx = idx[:8]
            elif self.user_args['feat_selection'] == 150:
                select_idx = idx[:11]
            elif self.user_args['feat_selection'] == 100:
                select_idx = idx[:15]
            elif self.user_args['feat_selection'] == 222:
                select_idx = idx[:9] + idx[11:]
                # [0.7672693681612002, 0.7792444976895054, 0.7943256845126212]
            elif self.user_args['feat_selection'] == 333:
                select_idx = idx[:8] + idx[8:]
                # [0.7858427339937082, 0.7572978824832088, 0.7844603685474677]
            self.data['features'] = pd.Series(np.array(self.data['features'].values.tolist())[:, select_idx].tolist())

        if self.is_training:
            return self.data[['qid', 'docno', 'features', 'label']]
        return self.data[['docno', 'features']]
 
    def _cal_embedding_dists(self):
        series = []
        distances = {
            'dot': lambda x, y : x.dot(y),
            'cos': lambda x, y : x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9),
            'euclidean': lambda x, y : np.linalg.norm(x - y)
        }
        for _, row in self.data.iterrows():
            cur_dists = []
            query_embedding = self.query_embeddings_dict[row['qid']]
            for field in FIELDS[:3]:
                if self.user_args['search_' + field]:
                    dists = [dist_func(query_embedding, row[field + '_embedding']) for dist_func in distances.values()]
                else:
                    dists = [np.nan] * len(distances)
                cur_dists.extend(dists)
            series.append(np.array(cur_dists))
        return pd.Series(series)

    def _pyterrier_rank(self):
        # field: title, abstract, subsection
        # method: bm25 tf-idf CoordinateMatch
        methods = ["TF_IDF", "BM25", "CoordinateMatch"]
        res_df = pd.DataFrame()
        for field in FIELDS[:-1]:
            if self.user_args['search_' + field]:
                pipeline =(
                    pt.text.scorer(body_attr=field, wmodel=methods[0])
                    **
                    pt.text.scorer(body_attr=field, wmodel=methods[1])
                    **
                    pt.text.scorer(body_attr=field, wmodel=methods[2])
                )
                res_df[field] = pipeline.transform(self.data)['features']
            else:
                res_df[field] = self.data.apply(lambda _: np.array([np.nan]*len(methods)), axis=1)
        res = res_df.apply(lambda x: np.concatenate(x[[0,1,2]]),axis = 1)
        return res

    def _get_doc_property(self):
        # publish time, conference, is_workshop, has_supp,
        pipeline = (
            (pt.apply.doc_score(lambda row: int(row["year"])))
            **
            (pt.apply.doc_score(lambda row: int(row["conference"]=="CVPR")))  #CVPR:1,ICCV:0
            **
            (pt.apply.doc_score(lambda row: int(row["workshop"]!='')))
            **
            (pt.apply.doc_score(lambda row: int(row["supp_link"]!='')))
            **
            (pt.apply.doc_score(lambda row: row['score']))
        )
        res = pipeline.transform(self.data)['features']
        return res

    def _match_author_name(self):
        # author hit-rate, try unigram and bi-gram
        def hit_rate(row):
            res = 0
            query = row['query'].split()
            authors = set(row['authors'].split(', '))
            for i in range(len(query)):
                if i != len(query)-1:
                    for author in authors:
                        #uni_hit
                        tmp = author.split()
                        if query[i] in tmp:
                            res+=1
                    #bi_hit
                    if (query[i]+" "+query[i+1]) in authors:
                        res+=1
                else:
                    for author in authors:
                        #uni_hit
                        tmp = author.split()
                        if query[i] in tmp:
                            res+=1
            return np.array([res])
        author_hit = self.data.apply(hit_rate, axis=1)
        return  author_hit


class PaperRetrieval():
    def __init__(self, index_root, wv_path, embedding_path, doc_path='./cleandatanew.pkl', 
                 json_path='', model_path='', args={}):
        # init: initialize pyterrier, load dataframes of embeddings and document texts ...
        self._init_pt()
        self.indexes = {}
        for field in FIELDS:
            index_rf = pjoin(index_root, field, 'data.properties')
            if not os.path.exists(index_rf):
                raise FileNotFoundError('Index missing! ' + index_rf)
            self.indexes[field] = pt.IndexFactory.of(index_rf)
        self.wv = KeyedVectors.load(wv_path, mmap='r')
        self.doc_embeddings = self._load_doc_embeddings(embedding_path)
        self.spelling_corrector = SpellingCorrector('./corpus.pkl')
        with open(doc_path, 'rb') as f:
            self.doc_text = pkl.load(f)
            self.doc_text['docno'] = self.doc_text['docno'].astype('str')
        self.const_user_args = {
            # define the default settings for our search engine filter
            'no_l2r': False,
            'recall_cutoff': 50,
            'search_title': True,
            'search_abstract': True,
            'search_subsections': True,
            'search_authors': True,
            'conference': None,
            'start_year': 0,
            'end_year': 9999,
            'must_have_supp': False,
            'recall_weights': [0.5, 0.4, 0.1, 1],
            'wmodel': 'BM25',
            'rerank': 'lgb'
        }
        self.const_user_args.update(args)
        self.user_args = copy.deepcopy(self.const_user_args)

        if not self.user_args['no_l2r']:
            if model_path != '' and os.path.exists(model_path):
                self._build_model(model_path)
            else:
                if self.user_args['rerank'] == 'lgb':
                    self.model = lgb.LGBMRanker(
                        task="train",
                        silent=True,
                        min_child_samples=1,
                        num_leaves=24,
                        max_depth=5,
                        objective="lambdarank",
                        metric="ndcg",
                        learning_rate= 0.064,
                        importance_type="gain",
                        num_iterations=150,
                        subsample=0.8
                    )
                elif self.user_args['rerank'] == 'logistic':
                    self.model = LogisticRegression(penalty='l2', C=1.0)
                elif self.user_args['rerank'] == 'lgbc':
                    self.model = lgb.LGBMClassifier(
                        max_depth=5,
                        learning_rate=0.6,
                        min_child_samples=1,
                        subsample=0.8,
                        max_leaves=24
                    )
                else:
                    raise ValueError('Wrong config for rerank model!')
                self._train(json_path, model_path)

    def _init_pt(self):
        if not pt.started():
            pt.init()
    
    def _do_recall(self, query):
        # get a list of related documents by BM25
        # return : DataFrame
        results = []
        for i, field in enumerate(FIELDS):
            if not self.user_args['search_' + field]:
                continue
            br = pt.BatchRetrieve(self.indexes[field], wmodel=self.user_args['wmodel'])
            result = br.transform(query)
            result['score'] *= self.user_args['recall_weights'][i]
            results.append(result)
        merged_scores = pd.concat(results).groupby(['qid', 'query', 'docno']).sum('score').reset_index()
        return merged_scores
        
    def _load_doc_embeddings(self, embedding_path):
        embed_df = pd.read_csv(embedding_path, sep=',')
        dim = self.wv.vector_size
        for i in range(1, 4):
            embed_df.iloc[:, i] = embed_df.iloc[:, i].apply(lambda s: np.array(s.split(), dtype=np.float64) if s != '0 . 0' else np.zeros(dim))
        embed_df['docno'] = embed_df['docno'].astype('str')
        return embed_df

    def _spelling_correct(self, query):
        original_words = re.findall(r'\w+', query.lower())
        corrections = [self.spelling_corrector.correction(w) for w in original_words]
        return ' '.join(corrections)

    def _recall_post_processing(self, recall_results):
        # filter invalid documents
        # take only top 50 documents for l2r
        # remember to join with the embeddings
        # time filter
        filter_idx1 = self.doc_text['year'] >= self.user_args['start_year']
        filter_idx2 = self.doc_text['year'] <= self.user_args['end_year']
        filter_idx = np.bitwise_and(filter_idx1, filter_idx2)
        # conference filter
        if self.user_args['conference'] is not None:
            filter_idx3 = self.doc_text['conference'].apply(lambda r: self.user_args['conference'] == r['conference'], axis=1)
            filter_idx = np.bitwise_and(filter_idx, filter_idx3)
        valid_text = self.doc_text.loc[filter_idx, 'docno']
        filtered_results = recall_results.merge(valid_text, how='inner', on='docno')
        sorted_results = filtered_results.sort_values(by='score', ascending=False).reset_index()
        cutoff_results = sorted_results[:self.user_args['recall_cutoff']]
        if not self.user_args['no_l2r']:
            results = cutoff_results.merge(self.doc_embeddings, how='left', on='docno')
            return results
        return cutoff_results

    def _get_training_data(self, json_root):
        # get the training query-doc pairs from the json files
        # remember to join with the embeddings
        train_file_names = os.listdir(pjoin(json_root, 'train'))
        val_file_names = os.listdir(pjoin(json_root, 'val'))
        train_pair_cnt = 0
        dfs = []
        for fname in train_file_names:
            json_df = pd.read_json(pjoin(json_root, 'train', fname))
            dfs.append(json_df)
            train_pair_cnt += len(json_df)
        for fname in val_file_names:
            json_df = pd.read_json(pjoin(json_root, 'val', fname))
            dfs.append(json_df)
        pairs = pd.concat(dfs).reset_index(drop=True)
        pairs['docno'] = pairs['docno'].astype('str')
        queries = list(pd.unique(pairs['query']))
        qids = {queries[qid-1]: qid for qid in range(1, len(queries)+1)}
        queries = pd.DataFrame({'qid': list(qids.values()), 'query': list(qids.keys())})
        base_df = self._do_recall(queries)
        data = pairs.merge(base_df, how='left', on=['docno', 'query'])
        data = data.merge(self.doc_embeddings, how='left', on='docno')
        return data[:train_pair_cnt], data[train_pair_cnt:]

    def _build_model(self, model_path):
        if self.user_args['rerank'] in ['lgb']:
            if not os.path.exists(model_path):
                if os.path.exists('./gbm_save.lgb'):
                    model_path = './gbm_save.lgb'
                else:
                    raise FileNotFoundError('Saved model is not found! ' + model_path)
            self.model = lgb.Booster(model_file=model_path)
        else:
            with open(model_path, 'rb') as f:
                self.model = pkl.load(f)
        print('Pretrained Model Loaded!')

    def _feature_normalize(self, features, stats_path='normed_features.npy', training=True):
        # features: numpy array [n, d]
        if training and os.path.exists(stats_path):
            stats = np.load(stats_path)
            mu, sigma = stats[0], stats[1]
        else:
            mu = np.mean(features, axis=0)
            sigma = np.std(features, axis=0)
            assert len(mu) == len(sigma)
            stats = np.concatenate([mu[None,:], sigma[None,:]], axis=0)
            np.save(stats_path, stats)
        sigma[sigma == 0] = 1
        features -= mu.reshape(1, -1)
        features /= sigma.reshape(1, -1)
        return features

    def _train(self, json_root, model_path):
        # load the training pair-doc pairs from the jsons
        if os.path.exists('./gbm_features.pkl'):
            with open('./gbm_features.pkl', 'rb') as f:
                feature_dic = pkl.load(f)
                train_features, val_features = feature_dic['train'], feature_dic['val']
        else:
            print('Start Getting Training Data!')
            train_data, val_data = self._get_training_data(json_root)
            print('Training Data Loaded!')
            train_data, val_data = self._merge_meta_data(train_data), self._merge_meta_data(val_data)
            train_fe = FeatureExtractor(train_data, self.indexes, self.wv, self.user_args, is_training=True)
            train_features = train_fe.get_features()
            print('Training Data Set Feature Extracted!')
            val_fe = FeatureExtractor(val_data, self.indexes, self.wv, self.user_args, is_training=True)
            val_features = val_fe.get_features()
            print('Validation Data Set Feature Extracted!')
            # with open('./gbm_features.pkl', 'wb') as f:
            #     feature_dic = {'train': train_features, 'val': val_features}
            #     pkl.dump(feature_dic, f)

        print('Start Training!')
        if self.user_args['rerank'] in ['lgb']:
            self.model.fit(np.array(train_features['features'].values.tolist()),
                        np.array(train_features['label'].values.tolist()),
                        group=train_features.groupby('qid')['qid'].count().to_numpy(),
                        eval_set=[
                            (np.array(val_features['features'].values.tolist()),
                                np.array(val_features['label'].values.tolist()))
                            ],
                        eval_group=[val_features.groupby('qid')['qid'].count().to_numpy()],
                        eval_at=[5, 10, 20],
                        eval_metric='ndcg'
            )
        else:
            train_feats = self._feature_normalize(np.array(train_features['features'].values.tolist()), training=True)
            val_feats = self._feature_normalize(np.array(val_features['features'].values.tolist()), training=False)
            train_label = (np.array(train_features['label'].values.tolist()) >= 3).astype('int32')
            val_label = np.array(val_features['label'].values.tolist())
            self.model.fit(train_feats, train_label)
            val_preds = self.model.predict_proba(val_feats)[:, 1]
            # import pdb; pdb.set_trace()
            groups = val_features.groupby('qid')['qid'].count().to_numpy()
            cur_g = 0
            val_ndcg = 0
            for cnt in groups:
                val_ndcg += ndcg_score(val_label[cur_g:cur_g+cnt].reshape(1,-1), val_preds[cur_g:cur_g+cnt].reshape(1,-1))
                cur_g += cnt
            print('validation ndcg: %.4f' % (val_ndcg / len(groups)))

        print('Done Training! Saving Model ... ')
        # with open(model_path, 'wb') as f:
        #     pkl.dump(self.model, f)
        if model_path:
            if self.user_args['rerank'] == 'lgb':
                self.model.booster_.save_model(filename=model_path, num_iteration=self.model.best_iteration_)
            elif self.user_args['rerank'] == 'logistic':
                with open(model_path, 'wb') as f:
                    pkl.dump(self.model, f)
        print('Done!')

    def _merge_meta_data(self, rank_results):
        return rank_results.merge(self.doc_text, how='left', on='docno')

    def search(self, query, args={}):
        # main function
        # do_recall: retrieve a list of relevant documents by BM25
        # recall_post_processing: filter out invalid docs according to user args
        # extract_features: build a feature extractor and return all features needed for the model
        # l2r inference: use l2r model to rank the documents
        if not isinstance(query, str):
            raise TypeError(f'Expected input type of str, get input {type(query)}')
        self.user_args = copy.deepcopy(self.const_user_args)
        self.user_args.update(args)
        query = self._spelling_correct(query)
        query_df = pd.DataFrame({'qid': [1], 'query': [query]})
        bm25_results = self._do_recall(query_df)
        try:
            recall_results = self._recall_post_processing(bm25_results)
        except KeyError:
            raise EmptyRetrievalError
        
        rank_results = self._merge_meta_data(recall_results)
        if not self.user_args['no_l2r']:
            feature_extractor = FeatureExtractor(rank_results, self.indexes, self.wv, self.user_args)
            feats = np.array(feature_extractor.get_features()['features'].tolist())
            if self.user_args['rerank'] not in ['lgb']:
                feats = self._feature_normalize(feats)
            preds = self.model.predict(feats)
            sort_idx = np.argsort(preds)[::-1].astype('int32')
            rank_results = rank_results.iloc[sort_idx, :].reset_index(drop=True)
        
        return rank_results[['docno', 'title', 'authors', 'abstract']]

if __name__ == '__main__':
    args = {
        'no_l2r': False,
        'rerank': 'lgb'
    }
    ir = PaperRetrieval(index_root='./index', wv_path='./word2vec_256.wv',
                        embedding_path='./word2vec_embedding_df_256.csv', json_path='./training_json/',
                        model_path='./gbm_256.model', args=args)
    test.test(ir)
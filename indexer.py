import os
import re
import numpy as np
import pandas as pd
import pyterrier as pt
import lightgbm as lgb
import pickle as pkl
from functools import reduce
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from os.path import join as pjoin
from spelling_corrector import SpellingCorrector


FIELDS = ['title', 'abstract', 'subsections', 'authors']

class FeatureExtractor():
    def __init__(self, df, indexes, word2vec, user_args={}, is_training=False):
        self.indexes = indexes
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
            hwords = reduce(lambda x, y: x + y, [[*word.split('-'), word.replace('-', '')] for word in hyphen_words])
            q_list = [term.lower() for term in query.split() + hwords]
            q_list = [ps.stem(term) for term in q_list]
            embedding = np.sum([word2vec[term] for term in q_list if term in word2vec.vocab], axis=0)
            if not isinstance(embedding, np.ndarray):
                embedding = np.zeros(word2vec.vector_size)
            return embedding

        q_dict = {qid: query for qid, query in self.data[['qid', 'query']].values}
        q_embed_dict = {qid: transform_query(query) for qid, query in q_dict.items()}
        return q_embed_dict

    def get_features(self):
        # missing values will be filled with np.nan
        # embeddings title, abstract, method 16d
        # title, abstract, method bm25 tf-idf CoordinateMatch
        # publish time, conference, is_workshop, has_supp, 
        # author hit-rate, try bi-gram
        feature_extractor_funcs = {
            # 'feature_name': self.func,
            'embedding_dists': self._cal_embedding_dists,
            'pyterrier_ranking': self._pyterrier_rank,
            'doc_property': self._get_doc_property,
            'author_name_match': self._match_author_name
        }

        self.data['features'] = self.data.apply(np.array([]))
        for feature_name in feature_extractor_funcs.keys():
            self.data['tmp'] = feature_extractor_funcs[feature_name]()
            self.data['features'] = self.data.apply(lambda x: np.concatenate(x['features'], x['tmp']), axis=1)
        
        if self.is_training:
            return self.data[['docno', 'features', 'label']]
        return self.data[['docno', 'features']]
 
    def _cal_embedding_dists(self):
        series = []
        distances = {
            'dot': lambda x, y : x.dot(y),
            'cos': lambda x, y : x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y),
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
        self.user_args = {
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
            'recall_weights': [0.5, 0.4, 0.1, 1]
        }
        self.user_args.update(args)

        if not self.user_args['no_l2r']:
            if model_path:
                self.model = self._build_model(model_path)
            else:
                self.model = lgb.LGBMRanker(
                    task="train",
                    silent=True,
                    min_child_samples=1,
                    num_leaves=31,
                    max_depth=5,
                    objective="lambdarank",
                    metric="ndcg",
                    learning_rate= 0.1,
                    importance_type="gain",
                    num_iterations=100
                )
                self._train(json_path)

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
            br = pt.BatchRetrieve(self.indexes[field], wmodel="BM25")
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
        filter_idx1 = self.doc_text['year'] >= self.user_args['start_year']
        filter_idx2 = self.doc_text['year'] <= self.user_args['end_year']
        filter_idx = np.bitwise_and(filter_idx1, filter_idx2)
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
        train_file_names = os.listdir(pjoin(json_root), 'train')
        val_file_names = os.listdir(pjoin(json_root), 'val')
        dfs = []
        for fname in train_file_names + val_file_names:
            json_df = pd.read_json(pjoin(json_root, fname))
            dfs.append(json_df)
        pairs = pd.concat(dfs, ignore_index=True)
        queries = pd.unique(pairs['query'])
        qids = {queries[qid]: qid for qid in range(1, len(queries)+1)}
        queries = pd.DataFrame({'qid': queries.apply(qids), 'query':queries})
        base_df = self._do_recall(queries)
        data = pairs.merge(base_df, how='left', on=['docno', 'query'])
        data = data.merge(self.doc_embeddings, how='left', on='docno')
        return data[: len(train_file_names)], data[len(train_file_names) :]

    def _build_model(self, model_path):
        if not os.path.exists(model_path):
            if os.path.exists('./gbm.save'):
                model_path = './gbm.save'
            else:
                raise FileNotFoundError('Saved model is not found! ' + model_path)
        else:
            self.model = lgb.load(model_path)

    def _train(self, json_root):
        # load the training pair-doc pairs from the jsons
        train_data, val_data = self._get_training_data(json_root)
        train_data, val_data = self._merge_meta_data(train_data), self._merge_meta_data(val_data)
        train_fe = FeatureExtractor(train_data, self.indexes, self.wv, self.user_args, True)
        train_features = train_fe.get_features()
        val_fe = FeatureExtractor(val_data, self.indexes, self.wv, self.user_args, True)
        val_features = val_fe.get_features()

        self.model.fit(train_features['features'],
                       train_features['label'],
                       group=train_features['qid'].groupby('qid')['qid'].count().to_numpy(),
                       eval_set=[(val_features['features'], val_features['label'])],
                       eval_group=val_features['qid'].groupby('qid')['qid'].count().to_numpy(),
                       eval_at=[10, 20, 50]
        )

        self.model.save_model('./gbm.save')

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
        self.user_args.update(args)
        query = self._spelling_correct(query)
        query_df = pd.DataFrame({'qid': [1], 'query': [query]})
        recall_results = self._recall_post_processing(self._do_recall(query_df))
        
        rank_results = self._merge_meta_data(recall_results)
        if not self.user_args['no_l2r']:
            feature_extractor = FeatureExtractor(rank_results, self.indexes, self.wv, self.user_args)
            feats = feature_extractor.get_features()
            preds = self.model.predict(feats)
            sort_idx = np.argsort(preds)[::-1]
            rank_results = rank_results[sort_idx].reset_index(drop=True)
        
        return rank_results[['docno', 'title', 'authors', 'abstract']]

if __name__ == '__main__':
    args = {
        'no_l2r': True
    }
    ir = PaperRetrieval(index_root='./index', wv_path='./word2vec.wv', embedding_path='./word2vec_embedding_df.csv', args=args)
    print(ir.search('batch normalization')[:10])
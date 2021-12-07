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
            title_embedding = row['title_embedding']
            abstract_embedding = row['abstract_embedding']
            subsections_embedding = row['subsections_embedding']
            query_embedding = self.query_embeddings_dict[row['qid']]
            if self.user_args['search_title']:
                title_dists = [dist_func(query_embedding, title_embedding) for dist_func in distances.values()]
            else:
                title_dists = [np.nan] * len(distances)
            if self.user_args['search_abstract']:
                abstract_dists = [dist_func(query_embedding, abstract_embedding) for dist_func in distances.values()]
            else:
                abstract_dists = [np.nan] * len(distances)
            if self.user_args['search_subsection']:
                subsection_dists = [dist_func(query_embedding, subsections_embedding) for dist_func in distances.values()]
            else:
                subsection_dists = [np.nan] * len(distances)
            series.append(np.array(title_dists + abstract_dists + subsection_dists))
        return pd.Series(series)



class PaperRetrieval():
    def __init__(self, index_root, wv_path, embedding_path, 
                 doc_path='./cleandatanew.pkl', json_path='', model_path=''):
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
        with open(doc_path, 'rb') as f:
            self.doc_text = pkl.load(f)
        self.user_args = {
            # define the default settings for our search engine filter
            'search_title': True,
            'search_abstract': True,
            'search_subsection': True,
            'conference': None,
            'year': None,
            'must_have_supp': False
        }
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

        if not json_path and not model_path:
            raise ValueError('Must indicate the path for training data or pretrained model')
        if model_path:
            self.model = self._build_model(model_path)
        else:
            self.model = self._train(json_path)

    def _init_pt(self):
        if not pt.started():
            pt.init()
    
    def _do_recall(self, query):
        # get a list of related documents by BM25
        # return : DataFrame
        bm25 = pt.BatchRetrieve(self.index, wmodel='BM25')
        return bm25.search(query)
        
    def _load_doc_embeddings(self, embedding_path):
        embed_df = pd.read_csv(embedding_path, sep=',')
        for i in range(1, 4):
            embed_df.iloc[:, i] = embed_df.iloc[:, i].apply(lambda s: np.array(s.split(), dtype=np.float64))
        embed_df['docno'] = embed_df['docno'].astype('str')
        return embed_df

    def _spelling_correct(self, query):
        pass

    def _recall_post_processing(self, recall_results):
        # remember to join with the embeddings
        pass

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
        qids = {queries[qid]: qid for qid in range(len(queries))}
        queries = pd.DataFrame({'qid': queries.apply(qids), 'query':queries})
        base_df = self._do_recall(queries)
        data = pairs.merge(base_df, how='left', on='docno')
        data = data.merge(self.doc_embeddings, how='left', on='docno')
        return data[: len(train_file_names)], data[len(train_file_names):]

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

    def search(self, query, args):
        # main function
        # do_recall: retrieve a list of relevant documents by BM25
        # recall_post_processing: filter out invalid docs according to user args
        # extract_features: build a feature extractor and return all features needed for the model
        # l2r inference: use l2r model to rank the documents
        if not isinstance(query, str):
            raise TypeError(f'Expected input type of str, get input {type(query)}')
        self.user_args.update(args)
        query = self._spelling_correct(query)
        query_df = pd.DataFrame({'qid': 1, 'query': query})
        recall_results = self._recall_post_processing(self._do_recall(query_df))
        
        feature_extractor = FeatureExtractor(recall_results, self.indexes, self.wv, self.user_args)
        feats = feature_extractor.get_features()
        preds = self.model.predict(feats)
        sort_idx = np.argsort(preds)[::-1]
        rank_results = recall_results[sort_idx].reset_index(drop=True)
        
        results = self._merge_meta_data(rank_results)
        return results['docno', 'title', 'authors', 'abstract']
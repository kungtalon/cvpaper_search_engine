import os
import re
import numpy as np
import pandas as pd
import pyterrier as pt
import lightgbm as lgb
from functools import reduce
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from os.path import join as pjoin


FIELDS = ['title', 'abstract', 'subsections', 'authors']

class FeatureExtractor():
    def __init__(self, df, indexes, embedding_path, wv_path, user_args={}, json_path='', is_training=False):
        self.indexes = indexes
        self.user_args = user_args
        if is_training:
            self.data = self._load_training_data(json_path, df)
        else:
            self.data = self._load_inference_data(df)
        self._load_doc_embeddings(embedding_path)
        self.query_embedding = self._get_query_embedding(wv_path)

    def _load_training_data(self, json_path, df):
        file_names = os.listdir(json_path)
        dfs = []
        for fname in file_names:
            json_df = pd.read_json(pjoin(json_path, fname))
            dfs.append(json_df)
        training_pairs = pd.concat(dfs, ignore_index=True)
        training_data = training_pairs.merge(df, how='left', on='docno')
        return training_data

    def _load_inference_data(self, df):
        pass

    def _load_doc_embeddings(self, embedding_path):
        pass

    def _get_query_embedding(self, wv_path)
        wv = KeyedVectors.load(wv_path, mmap='r')
        ps = PorterStemmer()
        # stop_words = set(stopwords.words('english'))
        query = self.data['query'][0]
        hyphen_words = re.findall('\w+-\w+', query)
        hwords = reduce(lambda x, y: x + y, [[*word.split('-'), word.replace('-', '')] for word in hyphen_words])
        q_list = [term.lower() for term in query.split() + hwords]
        q_list = [ps.stem(term) for term in q_list]
        embedding = np.sum([wv[term] for term in q_list if term in wv.vocab], axis=0)
        if not isinstance(embedding, np.ndarray):
            embedding = np.zeros(wv.vector_size)
        return embedding

    def get_features(self):
        # missing values will be filled with np.nan
        # embeddings title, abstract, method 16d
        # title, abstract, method bm25 tf-idf CoordinateMatch
        # publish time, conference, is_workshop, has_supp, 
        # author hit-rate, try bi-gram
        feature_extractor_funcs = {
            # 'feature_name': self.func,
            'embedding_dists': self.cal_embedding_dists,
            'put_embedding': self.put_embedding,
            'pyterrier_score': self.pyteriier_score,
            'doc_property': self.get_doc_property,
            'author_name_match': self.match_author_name
        }

        self.data['features'] = self.data.apply(np.array([]))
        for feature_name in feature_extractor_funcs.keys():
            self.data['tmp'] = feature_extractor_funcs[feature_name]()
            self.data['features'] = self.data.apply(lambda x: np.concatenate(x['features'], x['tmp']), axis=1)
        
        return self.data[['features', 'label']]

    def cal_embedding_dists(self):
        series = []
        distances = {
            'dot': lambda x, y : x.dot(y),
            'cos': lambda x, y : x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y),
            'euclidean': lambda x, y : np.linalg.norm(x - y)
        }
        for _, row in self.data.iterrows():
            title_embedding = row['title_embedding']
            abstract_embedding = row['abstract_embedding']
            subsection_embedding = row['subsection_embedding']
            if 'search_title' in self.user_args:
                title_dists = [dist_func(self.query_embedding, title_embedding) for dist_func in distances.values()]
            else:
                title_dists = [np.nan] * len(distances)
            if 'search_abstract' in self.user_args:
                abstract_dists = [dist_func(self.query_embedding, abstract_embedding) for dist_func in distances.values()]
            else:
                abstract_dists = [np.nan] * len(distances)
            if 'search_subsection' in self.user_args:
                subsection_dists = [dist_func(self.query_embedding, subsection_embedding) for dist_func in distances.values()]
            else:
                subsection_dists = [np.nan] * len(distances)
            series.append(np.array(title_dists + abstract_dists + subsection_dists))
        return pd.Series(series)



class CVPaperIR():
    def __init__(self, index_rf):
        self.init_pt()
        self.index = pt.Data
        self.bm

    def init_pt(self):
        if not pt.started():
            pt.init()
    
    def do_recall(self, query):
        # get a list of related documents by BM25
        bm25 = pt.BatchRetrieve(self.index, wmodel='BM25')
        return bm25.search(query)
        


    def recall_post_processing(self, ):
        pass

    def get_training_data(self, root):
        # get the training query-doc pairs from the json files
        # return: 
        pass

    def get_embedding(self, doc_id):
        # retrieve the sum-pooled embedding from dataframe
        pass


    def build_pipeline(self, recall_cutoff):
        # filter by fields
        # filter by year and conference
        # correction
        ltr_feats1 = (bm25 % recall_cutoff) >> pt.text.get_text(index, ["title", "date", "doi"]) >> (
            pt.transformer.IdentityTransformer()
            ** # sequential dependence
            (sdm >> bm25)
            **
            (qe >> bm25)
            ** # score of text for query 'coronavirus covid'
            (pt.apply.query(lambda row: 'coronavirus covid') >> bm25)
            ** # score of title (not originally indexed)
            (pt.text.scorer(body_attr="title", takes='docs', wmodel='BM25') ) 
            ** # date 2020
            (pt.apply.doc_score(lambda row: int("2020" in row["date"])))
            ** # has doi
            (pt.apply.doc_score(lambda row: int( row["doi"] is not None and len(row["doi"]) > 0) ))
            ** # abstract coordinate match
            pt.BatchRetrieve(index, wmodel="CoordinateMatch")
            # embeddings title, abstract, method 16d
            # title, abstract, method bm25 tf-idf CoordinateMatch
            # publish time, conference, is_workshop, has_supp, 
            # author hit-rate, try bi-gram
            # 
        )

        # for reference, lets record the feature names here too
        fnames=["BM25", "SDM", 'coronavirus covid', 'title', "2020", "hasDoi", "CoordinateMatch"]

        # acquire embeddings
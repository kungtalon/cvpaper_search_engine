import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
import pyterrier as pt
from os.path import join as pjoin


FIELDS = ['title', 'abstract', 'subsections']

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
            index_ref = pt.IndexRef(field_path)
        indexes[field] = pt.IndexFactory.of(index_ref)
    return indexes

def retrieve(query, indexes, weights):
    results = []
    for field in FIELDS:
        br = pt.BatchRetrieve(indexes[field], wmodel="BM25")
        result = br.search(query)[['paper_id', 'score']]
        result['score'] *= weights[field]
        results.append(results)
    import pdb; pdb.set_trace()
    return pd.concat(results).groupby('paper_id').sum()


def annotate(query_file):
    pass


def main():
    data = load_df_pickle('./cleandata.pkl')
    data['docno'] = data['paper_id'].astype('str')
    if not pt.started():
        pt.init()
    get_index_data(data, './index')

if __name__ == '__main__':
    main()
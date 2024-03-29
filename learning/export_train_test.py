import numpy as np
import pandas as pd
import pickle
import gc
from utils import filename


def load(name):
    print('Loading: {}'.format(name))
    d = None
    with open('output/unique_hash/{}.pickle'.format(name), 'rb') as f:
        d = pickle.load(f)
    return d


def export_all(hashes):
    n_df = pd.read_csv(filename.train_numeric_file, index_col=0, dtype=np.float32)
    for h in hashes:
        ids = items_with_response[h]
        rows = n_df.loc[ids].dropna(axis=1)
        export_filename = 'output/train_test/{}-{}.csv'.format('train_numeric', h)
        rows.to_csv(export_filename)
        print('.', end='', flush = True)
    del n_df
    gc.collect()

    d_df = pd.read_csv(filename.train_date_file, index_col=0, dtype=np.float32)
    for h in hashes:
        ids = items_with_response[h]
        rows = d_df.loc[ids].dropna(axis=1)
        export_filename = 'output/train_test/{}-{}.csv'.format('train_date', h)
        rows.to_csv(export_filename)
        print('.', end='', flush=True)
    del d_df
    gc.collect()

    c_df = pd.read_csv(filename.train_categorical_file, index_col=0, dtype=str)
    for h in hashes:
        ids = items_with_response[h]
        rows = c_df.loc[ids].dropna(axis=1)
        export_filename = 'output/train_test/{}-{}.csv'.format('train_categorical', h)
        rows.to_csv(export_filename)
        print('.', end='', flush=True)
    del c_df
    gc.collect()

    n_df = pd.read_csv(filename.test_numeric_file, index_col=0, dtype=np.float32)
    for h in hashes:
        ids = test_items_need_training[h]
        rows = n_df.loc[ids].dropna(axis=1)
        export_filename = 'output/train_test/{}-{}.csv'.format('test_numeric', h)
        rows.to_csv(export_filename)
        print('.', end='', flush=True)
    del n_df
    gc.collect()

    d_df = pd.read_csv(filename.test_date_file, index_col=0, dtype=np.float32)
    for h in hashes:
        ids = test_items_need_training[h]
        rows = d_df.loc[ids].dropna(axis=1)
        export_filename = 'output/train_test/{}-{}.csv'.format('test_date', h)
        rows.to_csv(export_filename)
        print('.', end='', flush=True)
    del d_df
    gc.collect()

    c_df = pd.read_csv(filename.test_categorical_file, index_col=0, dtype=str)
    for h in hashes:
        ids = test_items_need_training[h]
        rows = c_df.loc[ids].dropna(axis=1)
        export_filename = 'output/train_test/{}-{}.csv'.format('test_categorical', h)
        rows.to_csv(export_filename)
        print('.', end='', flush=True)
    del c_df
    gc.collect()

test_items_0 = load('test_items_0')
test_items_1 = load('test_items_1')
items_with_response = load('items_with_response')
test_items_need_training = load('test_items_need_training')

testing_set = set(list(test_items_need_training.keys()))
training_set = set(list(items_with_response.keys()))
intersection = training_set & testing_set
export_all(intersection)

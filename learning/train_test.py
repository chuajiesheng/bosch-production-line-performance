import numpy as np
import pandas as pd
import hashlib
import pickle
import gc
import csv
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from utils import filename
import code


def to_sha(s):
    m = hashlib.sha512()
    m.update(s.encode('utf-8'))
    return m.hexdigest()


def load(name):
    print('Loading: {}'.format(name))
    d = None
    with open('output/unique_hash/{}.pickle'.format(name), 'rb') as f:
        d = pickle.load(f)
    return d


def load_data(h, file_type, usecols=None, dtype=None):
    f = 'output/train_test/{}-{}.csv'.format(file_type, h)
    df = pd.read_csv(f, index_col=0, usecols=usecols, dtype=dtype)
    return df


def get_X(h, cols):
    n_df = load_data(h, 'train_numeric', usecols=cols, dtype=np.float32)
    d_df = load_data(h, 'train_date', dtype=np.float32)
    c_df = load_data(h, 'train_categorical', dtype=str)

    if c_df.shape[1] > 0:
        c_df = pd.get_dummies(c_df)
        print('c_df: {}'.format(c_df.dtypes.index))

    t_n_df = load_data(h, 'test_numeric', usecols=cols, dtype=np.float32)
    t_d_df = load_data(h, 'test_date', dtype=np.float32)
    t_c_df = load_data(h, 'test_categorical', dtype=str)

    if t_c_df.shape[1] > 0:
        t_c_df = pd.get_dummies(t_c_df)

    missing_columns = set(c_df.dtypes.index) - set(t_c_df.dtypes.index)
    for c in missing_columns:
        t_c_df[c] = 0
    t_c_df.sort_index(axis=1)

    missing_columns = set(t_c_df.dtypes.index) - set(c_df.dtypes.index)
    for c in missing_columns:
        c_df[c] = 0
    c_df.sort_index(axis=1)

    return np.concatenate([n_df, d_df, c_df], axis=1), np.concatenate([t_n_df, t_d_df, t_c_df], axis=1)

test_items_0 = load('test_items_0')
test_items_1 = load('test_items_1')
items_with_response = load('items_with_response')
test_items_need_training = load('test_items_need_training')

testing_set = set(list(test_items_need_training.keys()))
training_set = set(list(items_with_response.keys()))
intersection = training_set & testing_set

for h in intersection:
    print('hash: {}'.format(h))

    numeric_file = 'output/train_test/{}-{}.csv'.format('train_numeric', h)
    headers = None
    with open(numeric_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            headers = row
            break
    cols = len(headers)

    X, test_X = get_X(h, range(0, cols - 1))
    y = load_data(h, 'train_numeric', usecols=[0, cols - 1], dtype=np.float32).values.ravel()
    print('y: {}'.format(y.shape))

    base_score = float(y.sum()) / y.shape[0]
    clf = XGBClassifier(base_score=base_score)
    clf.fit(X, y)

    important_indices = np.where(clf.feature_importances_ > 0.0005)[0]
    print('important_indices: {}'.format(important_indices))

    folds = 3
    if y.shape[0] < folds:
        folds = y.shape[0]

    clf = XGBClassifier(max_depth=20, base_score=0.005)
    cv = StratifiedKFold(y, n_folds=folds)

    preds = np.ones(y.shape[0])
    for i, (train, test) in enumerate(cv):
        preds[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        if y.sum() != y.shape[0] and pd.Series(y[test]).unique().shape[0] > 1:
            print("Fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], preds[test])))
    print(roc_auc_score(y, preds))

    # pick the best threshold out-of-fold
    thresholds = np.linspace(0.01, 0.99, 50)
    mcc = np.array([matthews_corrcoef(y, preds > thr) for thr in thresholds])
    plt.plot(thresholds, mcc)
    best_threshold = thresholds[mcc.argmax()]
    print('mcc.max: {}'.format(mcc.max()))

    preds = (clf.predict_proba(test_X)[:, 1] > best_threshold).astype(np.int8)
    print('preds: {}'.format(preds))
    with open('output/test_pred/{}.pickle'.format(h), 'wb') as f:
        pickle.dump(preds, f, pickle.HIGHEST_PROTOCOL)

print('End')
import numpy as np
import pandas as pd


def read_csv_to_dataframe(filename, describe=False, fillna=True):
    df = pd.read_csv(filename, index_col=0, dtype=np.float32)

    if describe:
        df.describe()

    if fillna:
        df.fillna(0, inplace=True)

    return df


def features_to_dict(d, features):
    if d is None:
        d = dict()

    line = features[0]
    station = features[1]
    feature = features[2]

    if line not in d.keys():
        d[line] = dict()

    if station not in d[line].keys():
        d[line][station] = set()

    d[line][station].add(feature)
    return d


def get_feature_counts(headers):
    feature_set = dict()
    line_station_features = [s.split('_') for s in headers[1:]]

    for f in line_station_features:
        feature_set = features_to_dict(feature_set, f)

    for i in sorted(feature_set.keys()):
        for j in sorted(feature_set[i].keys()):
            print('Line {}, Station {} - {}'.format(i, j, len(feature_set[i][j])))

    return feature_set


def assert_feature_set(set_a, set_b):
    assert len(set_a.keys()) == len(set_b.keys())

    for i in sorted(set_a.keys()):
        assert i in set_b.keys()
        assert len(set_a[i].keys()) == len(set_b[i].keys())

        for j in sorted(set_a[i].keys()):
            assert j in set_b[i].keys()
            assert len(set_a[i][j]) == len(set_b[i][j])


numeric_datafile = 'input/train_numeric.csv'
date_datafile = 'input/train_date.csv'
categorical_datafile = 'input/train_categorical.csv'

date_df = read_csv_to_dataframe(date_datafile, fillna=False)
date_headers = list(date_df.columns.values)
date_feature_set = get_feature_counts(date_headers)

numeric_df = read_csv_to_dataframe(numeric_datafile, fillna=False)
numeric_headers = list(numeric_df.columns.values)
numeric_feature_set = get_feature_counts(numeric_headers)
assert_feature_set(date_feature_set, numeric_feature_set)

categorical_df = read_csv_to_dataframe(categorical_datafile, fillna=False)
categorical_headers = list(categorical_df.columns.values)
categorical_feature_set = get_feature_counts(categorical_headers)
assert_feature_set(date_feature_set, categorical_feature_set)
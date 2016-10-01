import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import code


def read_csv_to_dataframe(filename, describe=False, fillna=True):
    df = pd.read_csv(filename, index_col=0, dtype=np.float32)

    if describe:
        df.describe()

    if fillna:
        df.fillna(-1, inplace=True)

    return df

numeric_datafile = 'input/train_numeric.csv'
numeric_df = read_csv_to_dataframe(numeric_datafile, fillna=False)
numeric_headers = list(numeric_df.columns.values)

l0_features = []
l1_features = []
l2_features = []
l3_features = []

for f in numeric_headers:
    if f.startswith('L0'):
        l0_features.append(f)
    elif f.startswith('L1'):
        l1_features.append(f)
    elif f.startswith('L2'):
        l2_features.append(f)
    elif f.startswith('L3'):
        l3_features.append(f)

def check_either_this_or_that(features):
    idxs = [np.argwhere(feature_name == numeric_df.columns.values)[0][0] for feature_name in features]
    train_df = pd.read_csv(numeric_datafile,
                           index_col=0,
                           header=0,
                           usecols=[0, len(numeric_df.columns.values) - 1] + idxs)
    train_df[train_df.notnull()] = 1
    train_df[train_df.isnull()] = 0
    df_unique = train_df.drop_duplicates()
    df_unique = df_unique.T.drop_duplicates().T
    df_unique_headers = list(df_unique.columns.values)
    df_unique.sort(columns=df_unique_headers, inplace=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(df_unique, aspect='auto', cmap=plt.get_cmap('bwr'), interpolation='nearest')
    plt.show()  # plot the 1/0 map of the production line
    print ('# Finding column which are either this or that.')
    for current_column, next_column in list(zip(df_unique_headers, df_unique_headers[1:])):
        current_active = df_unique[current_column].sum()
        next_active = df_unique[next_column].sum()
        if current_active + next_active == df_unique[current_column].count():
            print('{} or {}'.format(current_column, next_column))

check_either_this_or_that(l0_features)
check_either_this_or_that(l1_features)
check_either_this_or_that(l2_features)

code.interact(local=dict(globals(), **locals()))

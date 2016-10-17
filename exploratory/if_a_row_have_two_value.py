import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import code
import hashlib


def to_md5(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()


def output_to_csv(k, rows):
    output_filename = 'output/{}.csv'.format(k)
    rows.to_csv(output_filename, mode='a')
    with open(output_filename, 'a') as f:
        f.write('\n')
        f.flush()

numeric_datafile = 'input/train_numeric.csv'
numeric_df = df = pd.read_csv(numeric_datafile, index_col=0, usecols=list(range(969)), dtype=np.float32)
numeric_df[numeric_df.notnull()] = 1
numeric_df.fillna(0, inplace=True)
# numeric_df is either 0 or 1

numeric_df['compressed'] = df.apply(lambda x: ''.join([str(int(v)) for v in x]), 1)
numeric_df.groupby('compressed').apply(lambda x: x.index.tolist())
grouped_by = numeric_df.groupby('compressed').apply(lambda x: x.index.tolist())

response_df = pd.read_csv("input/train_numeric.csv", index_col=0, usecols=[0, 969], dtype=np.float32)

unique_binary_string = 0
total_row = 0

binary_string_with_response = 0
true_for_all_row = 0

one_item_set = 0
one_item_set_true = 0

file_id_dict = dict()

for index, items in grouped_by.iteritems():
    total_row += len(items)
    unique_binary_string += 1

    responses = response_df.ix[items]
    total_response_value = responses.sum()['Response']

    if total_response_value > 0:
        binary_string_with_response += 1

    if total_response_value == len(items):
        if len(items) > 1:
            print('all {} items is true'.format(len(items)))
        true_for_all_row += 1

    if len(items) < 2:
        one_item_set += 1
        if total_response_value == 1:
            one_item_set_true += 1

    if total_response_value > 0 and 1 < len(items) < 100 and total_response_value != len(items):
        key = to_md5(index)
        file_id_dict[key] = list(map(lambda x: int(x), items))

print('{} true in {} one item only set'.format(one_item_set_true, one_item_set))
print('{} have true response in {} total binary string'.format(binary_string_with_response, unique_binary_string))
print('{} binary string have only true value'.format(true_for_all_row))

del numeric_df
del response_df

n_datafile = 'input/train_numeric.csv'
print('dumping from: {}'.format(n_datafile))
n_df = df = pd.read_csv(n_datafile, index_col=0, dtype=np.float32)
for k in file_id_dict.keys():
    print('dumping: {}'.format(file_id_dict[k]))
    output_to_csv(k, n_df.loc[file_id_dict[k]])
del n_df

d_datafile = 'input/train_date.csv'
print('dumping from: {}'.format(d_datafile))
d_df = df = pd.read_csv(d_datafile, index_col=0, dtype=np.float32)
for k in file_id_dict.keys():
    print('dumping: {}'.format(file_id_dict[k]))
    output_to_csv(k, d_df.loc[file_id_dict[k]])
del d_df

c_datafile = 'input/train_categorical.csv'
print('dumping from: {}'.format(c_datafile))
c_df = df = pd.read_csv(c_datafile, index_col=0, dtype=str)
for k in file_id_dict.keys():
    print('dumping: {}'.format(file_id_dict[k]))
    output_to_csv(k, c_df.loc[file_id_dict[k]])
del c_df

code.interact(local=dict(globals(), **locals()))

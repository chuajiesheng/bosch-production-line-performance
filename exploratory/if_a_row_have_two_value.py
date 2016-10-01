import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import code

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

print('{} true in {} one item only set'.format(one_item_set_true, one_item_set))
print('{} have true response in {} total binary string'.format(binary_string_with_response, unique_binary_string))
print('{} binary string have only true value'.format(true_for_all_row))

code.interact(local=dict(globals(), **locals()))

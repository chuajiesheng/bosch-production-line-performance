import numpy as np
import pandas as pd
import hashlib
import pickle
from utils import filename


def to_sha(s):
    m = hashlib.sha512()
    m.update(s.encode('utf-8'))
    return m.hexdigest()


def compress_to_path_string(f):
    df = pd.read_csv(f, index_col=0, usecols=list(range(969)), dtype=np.float32)

    df[df.notnull()] = 1
    df.fillna(0, inplace=True)
    # df is either 0 or 1

    df['compressed'] = df.apply(lambda x: ''.join([str(int(v)) for v in x]), 1)
    grouped = df.groupby('compressed').apply(lambda x: x.index.tolist())

    del df

    return grouped


def save(obj, name):
    print('Pickling: {}'.format(name))
    with open('output/unique_hash/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

train_numeric_groupings = compress_to_path_string(filename.train_numeric_file)

list_of_key_always_no = []
list_of_key_always_yes = []
items_with_response = dict()

response_df = pd.read_csv(filename.train_numeric_file, index_col=0, usecols=[0, 969], dtype=np.float32)
for k, indexes in train_numeric_groupings.iteritems():
    responses = response_df.ix[indexes]
    total_response_value = responses.sum()['Response']
    key = to_sha(k)

    if total_response_value == 0:
        list_of_key_always_no.append(key)
        continue

    if total_response_value == len(indexes):
        list_of_key_always_yes.append(key)
        continue

    items_with_response[key] = list(map(lambda x: int(x), indexes))

del response_df

test_items_0 = []
test_items_1 = []
test_items_need_training = dict()

test_numeric_groupings = compress_to_path_string(filename.test_numeric_file)
total_decided = 0
for k, indexes in test_numeric_groupings.iteritems():
    key = to_sha(k)

    if key in list_of_key_always_no:
        total_decided += len(indexes)
        test_items_0.extend(indexes)
        continue

    if key in list_of_key_always_yes:
        total_decided += len(indexes)
        test_items_1.extend(indexes)
        continue

    test_items_need_training[key] = list(map(lambda x: int(x), indexes))
del test_numeric_groupings

save(test_items_0, 'test_items_0')
save(test_items_1, 'test_items_1')
save(items_with_response, 'items_with_response')
save(test_items_need_training, 'test_items_need_training')

testing_set = set(list(test_items_need_training.keys()))
training_set = set(list(items_with_response.keys()))

all_hash = training_set | testing_set
intersection = training_set & testing_set
missing_in_training = all_hash - intersection

total_missing = 0
for h in missing_in_training:
    if h not in test_items_need_training.keys():
        continue
    test_ids = test_items_need_training[h]
    total_missing += len(test_ids)
print('total_missing: {}'.format(total_missing))


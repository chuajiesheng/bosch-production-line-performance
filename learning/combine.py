import pandas as pd
import pickle
import code


def load(name):
    print('## Loading: {}'.format(name))
    d = None
    with open('output/unique_hash/{}.pickle'.format(name), 'rb') as f:
        d = pickle.load(f)
    return d


def load_prediction(name):
    print('## Loading prediction: {}'.format(name))
    d = None
    with open('output/test_pred/{}.pickle'.format(name), 'rb') as f:
        d = pickle.load(f)
    return d

test_items_0 = load('test_items_0')
test_items_1 = load('test_items_1')
items_with_response = load('items_with_response')
test_items_need_training = load('test_items_need_training')

testing_set = set(list(test_items_need_training.keys()))
training_set = set(list(items_with_response.keys()))
intersection = training_set & testing_set

sub = pd.read_csv('input/sample_submission.csv', index_col=0)
print('sub shape: {}'.format(sub.shape))

for i in test_items_0:
    sub.set_value(i, 'Response', 0)
for i in test_items_1:
    sub.set_value(i, 'Response', 1)

total = len(test_items_0) + len(test_items_1)
for h in intersection:
    print('hash: {}'.format(h))
    try:
        preds = load_prediction(h)
    except FileNotFoundError:
        print('No records')
        continue

    assert len(test_items_need_training[h]) == len(preds)
    for index_of_preds, index_of_sub in enumerate(test_items_need_training[h]):
        sub.set_value(index_of_sub, 'Response', preds[index_of_preds])

    total += preds.shape[0]

print('{} of {} completed ({}%)'.format(total, sub.shape[0], round(total / sub.shape[0] * 100, 2)))

sub.to_csv("submission.csv.gz", compression="gzip")
print('End')

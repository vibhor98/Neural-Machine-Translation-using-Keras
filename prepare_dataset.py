import pickle as pkl
from numpy import random

def load_clean_data(filename):
    file = open(filename, 'rb')
    return pkl.load(file)

def save_clean_data(sentences, filename):
    pkl.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


raw_data = load_clean_data('english-german.pkl')

dataset = raw_data[:10000, :]
random.shuffle(dataset)

train_set = dataset[:9000, :]
test_set = dataset[9000:, :]

save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train_set, 'english-german-train.pkl')
save_clean_data(test_set, 'english-german-test.pkl')

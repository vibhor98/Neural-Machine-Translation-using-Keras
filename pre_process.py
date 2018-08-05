import string
import re
import pickle as pkl
import numpy as np
from unicodedata import normalize

# Load the file to preprocess
def load_file(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# Split the text into sentences
def to_pair(text):
    sentences = text.strip().split('\n')
    pairs = [s.strip().split('\t') for s in sentences]
    return pairs

# Clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # Regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            line = [re_print.sub('', word) for word in line]
            # Remove numeric chrs
            line = [w for w in line if w.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)

# Save the cleaned data to the given filename
def save_data(sentences, filename):
    pkl.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

filename = './data/deu.txt'
file = load_file(filename)
pairs = to_pair(file)
clean_pairs = clean_pairs(pairs)
save_data(clean_pairs, 'english-german.pkl')

# Checking the cleaned data
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))

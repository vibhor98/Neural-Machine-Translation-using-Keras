'''from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint


def load_dataset(filename):
    return load(open(filename, 'rb'))


# Fit a tokenizer
def create_tokenizer(lines):
    token = Tokenizer()
    token.fit_on_texts(lines)
    return token


def max_len_sent(lines):
    return max(len(line.split()) for line in lines)


# Encode and pad the input sequences
def encode_sequences(tokenizer, length, lines):
    # Embedding the input words in the sentence
    X = tokenizer.texts_to_sequences(lines)
    # Pad the encodings with zeros at the end
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# One-hot encoding of output sequence as the model will predict the expected
# word from the vocabulary
def encode_output(sequences, vocab_size):
    ylist = list()
    for seq in sequences:
        encoding = to_categorical(seq, num_classes=vocab_size)
        ylist.append(encoding)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# NMT model
def model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


# Load datasets
dataset = load_dataset('english-german-both.pkl')
train = load_dataset('english-german-train.pkl')
test = load_dataset('english-german-test.pkl')

# Prepare English tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_length = max_len_sent(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
print("English vocabulary size: %d" % eng_vocab_size)
print("Max length of English sentence: %d" % eng_length)

# Prepare German tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_length = max_len_sent(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
print("German vocabulary size: %d" % ger_vocab_size)
print("Max length of German sentence: %d" % ger_length)

# Prepare training data
trainX = encode_sequences(ger_tokenizer, ger_vocab_size, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_vocab_size, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

# Prepare test data
testX = encode_sequences(ger_tokenizer, ger_vocab_size, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_vocab_size, test[:, 0])
testY = encode_output(trainY, eng_vocab_size)

model = model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

# Fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)'''

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

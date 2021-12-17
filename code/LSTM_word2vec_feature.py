########################################
## import packages
########################################
import os
import re
import csv
import codecs
import time
import numpy as np
import pandas as pd

from string import punctuation
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from CNN1D import model_LSTM, model_LSTM_base

## set directories and parameters

BASE_DIR = './data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train_intsec.csv'
TEST_DATA_FILE = BASE_DIR + 'test_intsec.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True 

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

## index word vectors

print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print(('Found %s word vectors of word2vec' % len(word2vec)))


## process texts in datasets

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"What's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

train_1 = [] 
train_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        train_1.append(text_to_wordlist(values[3], remove_stopwords=False, stem_words=False))
        train_2.append(text_to_wordlist(values[4], remove_stopwords=False, stem_words=False))
        labels.append(int(values[5]))
print(('Found %s texts in train.csv' % len(train_1)))

test_1 = []
test_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_1.append(text_to_wordlist(values[1], remove_stopwords=False, stem_words=False))
        test_2.append(text_to_wordlist(values[2], remove_stopwords=False, stem_words=False))
        test_ids.append(values[0])
print(('Found %s texts in test.csv' % len(test_1)))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_1 + train_2 + test_1 + test_2)

train_sequences_1 = tokenizer.texts_to_sequences(train_1)
train_sequences_2 = tokenizer.texts_to_sequences(train_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_2)

word_index = tokenizer.word_index
print(('Found %s unique tokens' % len(word_index)))

train_data_1 = sequence.pad_sequences(train_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
train_data_2 = sequence.pad_sequences(train_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print(('Shape of data tensor:', train_data_1.shape))
print(('Shape of label tensor:', labels.shape))

test_data_1 = sequence.pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = sequence.pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

## prepare embeddings

print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in list(word_index.items()):
    if word in word2vec:
        embedding_matrix[i] = word2vec.word_vec(word)
print(('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)))


## sample train/validation data

perm = np.random.permutation(len(train_data_1))
idx_train = perm[:int(len(train_data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(train_data_1)*(1-VALIDATION_SPLIT)):]

train_df = pd.read_csv(TRAIN_DATA_FILE)
train_magic = train_df[['q1_count', 'q2_count', 'intersection_count']]
train_magic_bwd = train_df[['q2_count', 'q1_count', 'intersection_count']]
test_df = pd.read_csv(TEST_DATA_FILE)
test_magic_fwd = test_df[['q1_count', 'q2_count', 'intersection_count']]
test_magic_bwd = test_df[['q2_count', 'q1_count', 'intersection_count']]

ss = StandardScaler()
ss.fit(np.vstack((train_magic, test_magic_fwd)))
train_magic = ss.transform(train_magic)
test_magic_fwd = ss.transform(test_magic_fwd)

data_1_train = np.vstack((train_data_1[idx_train], train_data_2[idx_train]))
data_2_train = np.vstack((train_data_2[idx_train], train_data_1[idx_train]))
magic_train = np.vstack((train_magic[idx_train], train_magic[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((train_data_1[idx_val], train_data_2[idx_val]))
data_2_val = np.vstack((train_data_2[idx_val], train_data_1[idx_val]))
magic_val = np.vstack((train_magic[idx_val], train_magic[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))


ss = StandardScaler()
ss.fit(np.vstack((train_magic_bwd, test_magic_bwd)))
train_magic = ss.transform(train_magic_bwd)
test_magic_bwd = ss.transform(test_magic_bwd)

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344


## add class weight

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None


## define and run the model 

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

start_time = time.time()

model = model_LSTM(embedding_matrix)

hist = model.fit([data_1_train, data_2_train, magic_train], labels_train, \
        validation_data=([data_1_val, data_2_val, magic_val], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, verbose=1, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

print('Total running time is: ', time.time() - start_time)

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

preds = model.predict([test_data_1, test_data_2, test_magic_fwd], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, test_magic_bwd], batch_size=8192, verbose=1)
preds /= 2

result = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})

print('run baseline model')

start_time = time.time()

model = model_LSTM_base(embedding_matrix)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, verbose=1, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

print('Total running time is: ', time.time() - start_time)

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

result = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
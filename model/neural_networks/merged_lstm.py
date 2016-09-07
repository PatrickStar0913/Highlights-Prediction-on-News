# -*- coding: utf-8 -*-


from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Activation, Embedding, LSTM
import csv
import itertools
import re
import nltk
from keras.layers import recurrent
from keras.layers import Dense, Merge
import string
from sklearn.metrics import classification_report
from nltk.stem.lancaster import LancasterStemmer
import sys
csv.field_size_limit(sys.maxsize)

np.random.seed(1337)  # for reproducibility
wordEngStop = nltk.corpus.stopwords.words('english')
punctions = [',', '.', '!', ':', '@', ';', '#', '"', '?', '-', '/']
stopwords = punctions+wordEngStop
RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 32
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 5


def read_data(file1):

    with open(file1, 'r') as f:
        reader = csv.reader(f)
        reader.next()
        docs = [x[2] for x in reader]
    f.close()

    with open(file1, 'r') as f:
        reader = csv.reader(f)
        reader.next()
        sentences = [x[3] for x in reader]
    f.close()

    with open(file1, 'r') as f:
        reader = csv.reader(f)
        reader.next()
        values = [int(x[0]) for x in reader]
    f.close()

    # return docs, sentences, titles, values

    return docs, sentences, values


def check_digit(token):
    return all(c.isdigit() is False for c in token)


def tokenize(sent):
    lmtzr = LancasterStemmer()
    vectors = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    # remove stopwords
    new_vectors = [x for x in vectors if x not in stopwords]
    exclude = set(string.punctuation)
    # remove punctuations
    new_vectors = [''.join(ch for ch in x if ch not in exclude) for x in new_vectors]
    new_vectors = [lmtzr.stem(x) for x in new_vectors]
    # if the word including digits, then change it to a specific character
    new_vectors = ['*' if check_digit(x) is False else x for x in new_vectors]
    # print(len(new_vectors))
    if len(new_vectors) > 5000:
        new_vectors = new_vectors[0:4999]

    return new_vectors


def tokenize_sen(sentences):
    tokenized_sentences = [tokenize(sent) for sent in sentences]
    return tokenized_sentences


def vectorize_sentences(sentences, titles, values, word_idx, story_maxlen, query_maxlen):
    X = []
    Xt = []
    for sen in sentences:
        x = [word_idx[w] for w in sen]
        X.append(x)
    for title in titles:
        t = [word_idx[w] for w in title]
        Xt.append(t)
    # tansform the 2-dimensional into 3-D
    return sequence.pad_sequences(X, maxlen=story_maxlen), sequence.pad_sequences(Xt, maxlen=query_maxlen), np.array(values)


def prepare_predicts(x_test_a):

    predict = model.predict(x_test_a, batch_size = 1)

    predicts = []
    for i in predict:
        if i > 0.5:
            predicts.append(1)
        else:
            predicts.append(0)

    return predicts

if __name__ == "__main__":


    # read data
    print("Reading CSV file...")
    docs, sens, values = read_data('../data/output/news_gra_sen_title_train.csv')
    docs_test, sens_test, values_test = read_data('../data/output/news_gra_sen_title_test.csv')

    # tokenize sentence and title
    docs = tokenize_sen(docs)
    sens = tokenize_sen(sens)
    docs_test = tokenize_sen(docs_test)
    sens_test = tokenize_sen(sens_test)

    # flat lists into one list
    flatten_docs = [y for x in docs for y in x]
    flatten_sens = [y for x in sens for y in x]
    flatten_docs_test = [y for x in docs_test for y in x]
    flatten_sens_test = [y for x in sens_test for y in x]

    # build the dictionary
    vocab = sorted(set(flatten_docs + flatten_docs_test + flatten_sens + flatten_sens_test))
    # print vocab
    vocab_size = len(vocab) + 1
    print("vs",vocab_size)

    # use hashmap to store the dictionary
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    # print word_idx

    # find the maximum length,used in input_shape of lstm
    doc_maxlen = max(max(map(len, (x for x in docs))),max(map(len, (x for x in docs_test))))
    sen_maxlen = max(max(map(len, (x for x in sens))),max(map(len, (x for x in sens_test))))
    print(doc_maxlen)
    print(sen_maxlen)
    # maxlen = max(doc_maxlen, sen_maxlen)


    x_train_a, x_train_b, y_train = vectorize_sentences(docs, sens, values, word_idx, doc_maxlen, sen_maxlen)
    x_test_a, x_test_b, y_test = vectorize_sentences(docs_test, sens_test, values_test, word_idx, doc_maxlen, sen_maxlen)
    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(x_train_a.shape))
    print('Xq.shape = {}'.format(x_train_b.shape))
    print('Y.shape = {}'.format(y_train.shape))
    print('Y_test.shape = {}'.format(y_test.shape))

    print('Build model...')

    # build the model
    encoder_a = Sequential()
    encoder_a.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=doc_maxlen, dropout=0.1))
    encoder_a.add(LSTM(EMBED_HIDDEN_SIZE, dropout_W=0.1, dropout_U=0.1))

    encoder_b = Sequential()
    encoder_b.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=sen_maxlen, dropout=0.1))
    encoder_b.add(LSTM(EMBED_HIDDEN_SIZE, dropout_W=0.1, dropout_U=0.1))

    model = Sequential()
    model.add(Merge([encoder_a, encoder_b], mode='sum'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print('Training')
    model.fit([x_train_a, x_train_b], y_train, batch_size=BATCH_SIZE, nb_epoch=4, validation_split=0.01)


    predicts = prepare_predicts([x_test_a,x_test_b])
    print(predicts)
    print(y_test)
    report = classification_report(predicts, y_test, target_names=['Negative', 'Positive'])
    print('=' * 80)
    print("Accuracy Report Table: \n\n", report)


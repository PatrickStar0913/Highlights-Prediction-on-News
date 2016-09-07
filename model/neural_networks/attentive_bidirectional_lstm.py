
from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.layers import Embedding, LSTM
import csv
import sys
import re
import nltk
from keras.layers import recurrent
import string
from sklearn.metrics import classification_report
from nltk.stem.lancaster import LancasterStemmer
from keras.engine import Input, merge
from model.custom import Reverse, masked_concat, masked_dot, masked_sum, MaskedFlatten
from keras.layers.core import Activation, Dense, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.callbacks import EarlyStopping



csv.field_size_limit(sys.maxsize)

np.random.seed(1337)  # for reproducibility
wordEngStop = nltk.corpus.stopwords.words('english')
punctions = [',', '.', '!', ':', '@', ';', '#', '"', '?', '-', '/','',' ','  ','   ','    ','      ','       ']
stopwords = punctions+wordEngStop
RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 32
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
        # Split full comments into sentences
        titles = [x[4] for x in reader]
    f.close()

    with open(file1, 'r') as f:
        reader = csv.reader(f)
        reader.next()
        values = [int(x[0]) for x in reader]
    f.close()

    return docs, sentences, titles, values


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


def vectorize_sentences(docs, sentences, titles, values, word_idx, doc_maxlen, sen_maxlen ,title_maxlen):
    X_d = []
    X = []
    Xt = []
    for doc in docs:
        x_d = [word_idx[w] for w in doc]
        X_d.append(x_d)
    for sen in sentences:
        x = [word_idx[w] for w in sen]
        X.append(x)
    for title in titles:
        t = [word_idx[w] for w in title]
        Xt.append(t)
    # tansform the 2-dimensional into 3-D
    return sequence.pad_sequences(X_d, maxlen=doc_maxlen),sequence.pad_sequences(X, maxlen=sen_maxlen), sequence.pad_sequences(Xt, maxlen=title_maxlen), np.array(values)


def prepare_predicts(x_test_a):

    predict = model.predict(x_test_a, batch_size=1)
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
    docs, sens, titles, values = read_data('../data/output/news_gra_sen_title_train.csv')
    print(docs[0])
    print(sens[0])
    print(titles[0])
    print(values[0])
    docs_test, sens_test, titles_test, values_test = read_data('../data/output/news_gra_sen_title_test.csv')

    print("read successfully")
    # print(titles)
    # story_maxlen = 5000


    # tokenize sentence and title
    docs = tokenize_sen(docs)
    sens = tokenize_sen(sens)
    titles = tokenize_sen(titles)

    docs_test = tokenize_sen(docs_test)
    sens_test = tokenize_sen(sens_test)
    titles_test = tokenize_sen(titles_test)

    print("tokenize successfully")

    # flat lists into one list
    flatten_docs = [y for x in docs for y in x]
    flatten_sens = [y for x in sens for y in x]
    flatten_titles = [y for x in titles for y in x]
    print("flatten 1")

    flatten_docs_test = [y for x in docs_test for y in x]
    flatten_sens_test = [y for x in sens_test for y in x]
    flatten_titles_test = [y for x in titles_test for y in x]

    print("flatten successfully")

    # build the dictionary
    vocab = sorted(set(flatten_docs + flatten_docs_test + flatten_sens + flatten_sens_test+ flatten_titles + flatten_titles_test))
    # print vocab
    vocab_size = len(vocab) + 1
    print("vs",vocab_size)

    # use hashmap to store the dictionary
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    # print word_idx

    # find the maximum length,used in input_shape of lstm
    doc_maxlen = max(max(map(len, (x for x in docs))),max(map(len, (x for x in docs_test))))
    sen_maxlen = max(max(map(len, (x for x in sens))),max(map(len, (x for x in sens_test))))
    title_maxlen = max(max(map(len, (x for x in titles))),max(map(len, (x for x in titles_test))))
    # query_maxlen = 5000
    # maxlen = max(story_maxlen, query_maxlen)


    x_train_a, x_train_b,x_train_c,y_train = vectorize_sentences(docs, sens, titles, values, word_idx, doc_maxlen, sen_maxlen, title_maxlen)
    x_test_a, x_test_b,x_test_c, y_test = vectorize_sentences(docs_test,sens_test, titles_test, values_test, word_idx, doc_maxlen, sen_maxlen, title_maxlen)
    print('vocab = {}'.format(vocab))
    print('X_d.shape = {}'.format(x_train_a.shape))
    print('X.shape = {}'.format(x_train_b.shape))
    print('Xq.shape = {}'.format(x_train_c.shape))
    print('Y.shape = {}'.format(y_train.shape))
    print('Y_test.shape = {}'.format(y_test.shape))

    print('Build model...')

    # build the model
    embed_weights = np.zeros((vocab_size + 2, EMBED_HIDDEN_SIZE), dtype='float32')
    word_dim = embed_weights.shape[1]
    print("word_dim:  ", word_dim)

    doc_input = Input(shape=(doc_maxlen,), dtype='int32', name="DocInput")
    x = Embedding(input_dim=vocab_size+2,
                  output_dim=word_dim,
                  input_length=doc_maxlen,
                  mask_zero=True,
                  weights=[embed_weights])(doc_input)
    doc_lstm_f = LSTM(EMBED_HIDDEN_SIZE,
                        return_sequences = True,
                        consume_less='gpu',)(x)
    doc_lstm_b = LSTM(EMBED_HIDDEN_SIZE,
                        return_sequences = True,
                        consume_less='gpu',
                        go_backwards=True,)(x)
    doc_lstm_b_r = Reverse()(doc_lstm_b)
    yd = masked_concat([doc_lstm_f, doc_lstm_b_r])

    sen_input = Input(shape=(sen_maxlen,), dtype='int32', name='SenInput')
    x_q = Embedding(input_dim=vocab_size+2,
            output_dim=word_dim,
            input_length=sen_maxlen,
            mask_zero=True,
            weights=[embed_weights])(sen_input)
    sen_lstm_f = LSTM(EMBED_HIDDEN_SIZE,
                        consume_less='gpu',)(x_q)
    sen_lstm_b = LSTM(EMBED_HIDDEN_SIZE,
                        go_backwards=True,
                        consume_less='gpu',)(x_q)
    u = merge([sen_lstm_f, sen_lstm_b], mode='concat')

    title_input = Input(shape=(title_maxlen,), dtype='int32', name='TitleInput')
    x_t = Embedding(input_dim=vocab_size+2,
            output_dim=word_dim,
            input_length=title_maxlen,
            mask_zero=True,
            weights=[embed_weights])(title_input)
    title_lstm_f = LSTM(EMBED_HIDDEN_SIZE,
                        consume_less='gpu',)(x_t)
    title_lstm_b = LSTM(EMBED_HIDDEN_SIZE,
                        go_backwards=True,
                        consume_less='gpu',)(x_t)
    u_title = merge([title_lstm_f, title_lstm_b], mode='concat')

    u_rpt = RepeatVector(doc_maxlen)(u_title)
    story_dense = TimeDistributed(Dense(2*EMBED_HIDDEN_SIZE))(yd)
    title_dense = TimeDistributed(Dense(2*EMBED_HIDDEN_SIZE))(u_rpt)
    story_query_sum = masked_sum([story_dense, title_dense])

    m = Activation('tanh')(story_query_sum)
    w_m = TimeDistributed(Dense(1))(m)
    w_m_flat = MaskedFlatten()(w_m)
    s = Activation('softmax')(w_m_flat)
    r = masked_dot([s, yd])
    g_r = Dense(word_dim)(r)
    g_u = Dense(word_dim)(u)
    g_r_plus_g_u = merge([g_r, g_u], mode='sum')
    g_d_q = Activation('tanh')(g_r_plus_g_u)
    # g_d_q = Activation('tanh')(g_r)
    # result1 = LSTM(EMBED_HIDDEN_SIZE,
    #                     consume_less='gpu',)(g_d_q)
    result = Dense(1, activation='sigmoid')(g_d_q)

    model = Model(input=[doc_input,sen_input,title_input], output=result)

    # keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.02, decay=1e-6)
    # RMSprop = RMSprop(lr=0.000001, rho=0.9, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    print('Training')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit([x_train_a,x_train_b,x_train_c], y_train, batch_size=BATCH_SIZE, nb_epoch=4, validation_split=0.01, callbacks=[early_stopping])
    # model.fit(x_train_a, y_train, batch_size=BATCH_SIZE, nb_epoch=100, validation_split=0.05)
    # model.fit([x_train_a, x_train_b], y_train, batch_size=BATCH_SIZE, nb_epoch=50)
    # loss, acc = model.evaluate([x_test_a,x_test_b], y_test, batch_size=BATCH_SIZE)

    predicts = prepare_predicts([x_test_a,x_test_b,x_test_c])
    # predicts = model.predict(x_test_a, batch_size = 1)
    # print(predicts)
    # print(y_test)
    report = classification_report(predicts, y_test, target_names=['Negative', 'Positive'])
    print('=' * 80)
    print("Accuracy Report Table: \n\n", report)


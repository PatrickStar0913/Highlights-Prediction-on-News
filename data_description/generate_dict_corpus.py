from gensim import corpora, models, similarities
from nltk.corpus import stopwords
import string
from collections import defaultdict
from nltk.stem import SnowballStemmer
import csv

stop_words = stopwords.words('english') + ['one','two','three','four','five','six','seven','eight','nine','ten','first','second','third',\
'cant','get','another','still','wasnt','werent']

def read_file(file):
    documents = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        reader.next()
        selections = [x[3] for x in reader]
        # sentences = ["%s" % (x) for x in sentences]
    f.close()

    # remove non_english
    for selection in selections:
        selection = selection.decode("utf8")
        # print selection
        remove_non_ascii(selection)
        selection = selection.encode("ascii", "ignore")
        documents.append(selection)
    return documents


def is_ascii(s):
     return all(ord(c) < 128 for c in s)


def remove_punctions(documents):
    # remove punctuations and low words
    tokens = [[word.translate(None, string.punctuation).lower() for word in doc.split()]
              for doc in documents]
    return tokens


def remove_digits(documents):
    tokens = [[word.translate(None, string.digits).lower() for word in doc]
              for doc in documents]
    return tokens


def remove_unfrequent(documents):
    # remove words only appearing once
    frequency = defaultdict(int)
    for doc in documents:
        for token in doc:
             frequency[token] += 1

    texts = [[token for token in doc if frequency[token] > 1]
              for doc in documents]
    return texts


def remove_lessthan3letters(documents):
    tokens = [[word for word in doc if len(word)>3]
              for doc in documents]
    return tokens


def remove_non_ascii(text):
        return ' '.join(i for i in text if 0 < ord(i) < 127)


# remove stopwords
def get_filter(documents):
    filtered = [[w for w in doc if w not in stop_words]
                for doc in documents]
    return filtered


def get_stemmed(filtered):
    stemmed= [[SnowballStemmer('english').stem(w) for w in token]
                for token in filtered]
    return stemmed


def get_prepared_docs():

    dictionary = corpora.dictionary.Dictionary.load('../data/temporary/dict_highlights.dict')
    return dictionary


if __name__ == '__main__':

    # read data
    file = '../data/output/news_gra_sen_title.csv'
    # remove non English
    documents = read_file(file)

    # remove non punctions
    doc_non_punctions = remove_punctions(documents)

    # remove digits
    doc_non_digits = remove_digits(doc_non_punctions)



    # remove words only appears once
    doc_frequent= remove_unfrequent(doc_non_digits)

    # remove words whose length is less than 3
    doc_morethan3 = remove_lessthan3letters(doc_frequent)

    # stem them
    doc_stemmed = get_stemmed(doc_morethan3)
    # print doc_stemmed

    # remove stop words
    doc_filtered = get_filter(doc_stemmed)
    #
    print "generating dictionary..."
    dictionary = corpora.Dictionary(doc_filtered)
    print "dictionary generated successfully!"
    dictionary.save('../data/temporary/dict.dict')
    print dictionary
    # print dictionary


    # print "loading dictionary..."
    # dictionary = get_prepared_docs()
    # print "dictionary loaded successfully!"

    print "generating corpus vectors..."
    corpus = [dictionary.doc2bow(doc) for doc in doc_filtered]
    corpora.MmCorpus.serialize('../data/temporary/corpus.mm', corpus)
    print "corpus vectors generated successfully!"

    # print "loading corpus vectors..."
    # corpus = corpora.MmCorpus('corpus_highlight.mm')
    # corpora.MmCorpus.serialize('corpus_highlight.mm', corpus)
    # print "corpus vectors loaded successfully!"

    # print(corpus)
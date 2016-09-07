from gensim import corpora, models, similarities
import os
from gensim.models.coherencemodel import CoherenceModel

# load dictionary and corpus
if (os.path.exists('../data/temporary/dict.dict')):
    dictionary = corpora.Dictionary.load('../data/temporary/dict.dict')
    corpus = corpora.MmCorpus('../data/temporary/corpus.mm')
    print("Used files generated from first tutorial")
    print corpus
else:
    print("Please run first tutorial to generate data set")

# use tfidf to prepare corpus
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# generate the topic model
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, iterations=200, num_topics=30)
doc_lda = lda[corpus_tfidf]
lda.save('../data/temporary/model.lda')
print lda.print_topics(num_topics=30, num_words=30)

# sort topics by the accumulated distribution
count = []
for i in range(0, 30):
    count.append(0)

for doc in doc_lda:
    for token in doc:
        tag = int(token[0])
        count[tag] += float(token[1])

for i in range(0, 30):
    print count[i], "   ", i
print sorted(count,reverse=True)
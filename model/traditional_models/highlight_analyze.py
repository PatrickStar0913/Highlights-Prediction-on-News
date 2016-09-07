from sklearn.cross_validation import train_test_split
from scipy import sparse
import numpy as np
from data_process import DataProcess
from feature_extraction import FeatureExtraction
from evaluation.svm import EvalSVM
from evaluation.tree import EvalTree
from evaluation.nb import EvalNB
from evaluation.logReg import EvalLogReg
from sklearn.decomposition import PCA

# Init the data process and feature extraction object
data_process = DataProcess()
feature_extraction = FeatureExtraction()


# data_content, data_lable, data_similarity = data_process.load_data('2w_sample_sim.csv')
data_content, data_label,data_similarity,data_b1,data_b2,data_b3,data_b4,data_b5 = data_process.load_data('../../data/output/news_gra_sen_title_sample_sim.csv')
# print distribution
data_process.extract_n_p_total(data_label)
# print data_content
# print data_similarity
# print data_b5

# preapre features, similarity, location
data_similarity = np.array(data_similarity)
# data_b1 = np.array(data_b1)
# data_b2 = np.array(data_b2)
# data_b3 = np.array(data_b3)
# data_b4 = np.array(data_b4)
# data_b5 = np.array(data_b5)
metadata1 =sparse.csr_matrix([data_similarity.astype(float)]).T
# metadata2 =sparse.csr_matrix([data_b1.astype(float)]).T
# metadata3 =sparse.csr_matrix([data_b2.astype(float)]).T
# metadata4 =sparse.csr_matrix([data_b3.astype(float)]).T
# metadata5 =sparse.csr_matrix([data_b4.astype(float)]).T
# metadata6 =sparse.csr_matrix([data_b5.astype(float)]).T

# process data
train_processed_data = data_process.pre_process(data_content)
train_processed_data = data_process.lemmatizer(train_processed_data)

# vectorize sentence to sparse vectors
train_vectorized_data = feature_extraction.tfidf_vectorizer(train_processed_data)


# add features, similarity, location
allData = sparse.hstack([train_vectorized_data, metadata1]).tocsr()
# allData = sparse.hstack([allData, metadata2]).tocsr()
# allData = sparse.hstack([allData, metadata3]).tocsr()
# allData = sparse.hstack([allData, metadata4]).tocsr()
# allData = sparse.hstack([allData, metadata5]).tocsr()
# allData = sparse.hstack([allData, metadata6]).tocsr()
# print allData.shape
# allData = allData.toarray()
# pca = PCA(n_components=30)
# allData = pca.fit_transform(allData)

print allData.shape

# split data
a_train, a_test, b_train, b_test = train_test_split(train_vectorized_data, data_label, test_size=0.2, random_state=42)

# init classifier
# classifier = EvalSVM(1, 100)
# classifier = EvalTree()
classifier = EvalLogReg()
clf = classifier.init_classifier()
# train data
clf = classifier.fit_train_data(clf, a_train, b_train)
print "trained successfully"

# print result
classifier.eval_output(clf, a_train, b_train, a_test, b_test)
classifier.accuracy(b_test)

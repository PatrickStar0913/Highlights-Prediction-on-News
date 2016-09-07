# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, cross_validation
from pprint import pprint
from datetime import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import style
from nltk.stem.lancaster import LancasterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import os
import re
import random
import pandas as pd
import numpy as np
import itertools
import re
# Useing MatLab like graph style
style.use('ggplot')


class EvalLogReg(object):
    """docstring for EvalTree"""
    b_preict = []
    def __init__(self):
        super(EvalLogReg, self).__init__()




    def init_classifier(self):
        # clf = svm.SVC(kernel = 'rbf', gamma=self.gamma_value, C=self.c_value)
        # print "SVM configuration... \n\n", clf
        # clf = LogisticRegression()
        clf = SGDClassifier(loss="hinge", penalty="l2")
        return clf



    def fit_train_data(self, clf, a_train, b_train):
        # clf = svm.SVC(kernel = 'rbf', gamma=gamma_value, C=c_value)
        # print "SVM configuration... \n\n", clf
        print('=' * 50)
        clf.fit(a_train, b_train)
        return clf


    def eval_output(self, clf, a_train, b_train, a_test, b_test):

        b_predict = clf.predict(a_test)
        print b_predict
        self.b_predict = b_predict
        print "Number of %d has been predicted" % len(b_predict), '\n\n'

        print "The results shows below"


        scores = cross_validation.cross_val_score(clf, a_train, b_train, cv=10)
        print('=' * 80)
        print "Cross Validation is \n",scores
        print("Mean Accuracy of Cross Validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        report = classification_report(b_test, b_predict, target_names = ['Negative', 'Positive'])
        print('=' * 80)

        print "Accuracy Report Table: \n\n", report


    def accuracy(self, test_label):

        predict_label = self.b_predict

        positive = 1
        negative = 0

        pos_count = 0
        neg_count = 0

        for i, j in zip(predict_label, test_label):

            if i == "1" and j == "1":
                pos_count += 1

            if i == "0" and j == "0":
                neg_count += 1

        sample_sum = len(predict_label)

        print "pos sum",pos_count, "neg sum", neg_count

        accuracy = float(pos_count + neg_count) / sample_sum
        print accuracy

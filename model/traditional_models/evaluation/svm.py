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
# from matplotlib import style
from nltk.stem.lancaster import LancasterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import os
import re
import random
import pandas as pd
import numpy as np
import itertools
import re
# Useing MatLab like graph style
# style.use('ggplot')


class EvalSVM(object):
    """docstring for EvalSVM"""
    b_preict = []
    def __init__(self, gamma, c):
        super(EvalSVM, self).__init__()
        self.gamma_value = gamma
        self.c_value = c


    def init_classifier(self):
        clf = svm.SVC(kernel = 'rbf', gamma=self.gamma_value, C=self.c_value)
        # print "SVM configuration... \n\n", clf
        # clf = MultinomialNB()
        return clf



    def fit_train_data(self, clf, a_train, b_train):
        # clf = svm.SVC(kernel = 'rbf', gamma=gamma_value, C=c_value)
        # print "SVM configuration... \n\n", clf
        print('=' * 50)
        clf.fit(a_train, b_train)
        return clf


    def eval_output(self, clf, a_train, b_train, a_test, b_test):

        b_predict = clf.predict(a_test)
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


    """
    Grid Search Section
    Exhausted search of predefined parameter
    """
    def parameter_turning(self, a_train, b_train):


        print('=' * 80)
        print "Grid Seach For Best Estimator"

        parameters = {'C':(0.2,0.5,1,2,3,4,5,10),
                      'gamma':(0.2,0.5,1,2,3,4,5,10)}

        C_range = 10. ** np.arange(-2, 9)
        gamma_range = 10. ** np.arange(-5, 4)

        param_grid = dict(gamma=gamma_range, C=C_range)


        gs_clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, n_jobs=-1)

        # Fit and train the train data
        gs_clf = gs_clf.fit(a_train,b_train)
        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

        # Print the score for each parameters
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print "Score is "
        print score


        print("The best classifier is: ", gs_clf.best_estimator_)


        # plot the scores of the grid
        # grid_scores_ contains parameter settings and scores
        score_dict = gs_clf.grid_scores_

        # We extract just the scores
        scores = [x[1] for x in score_dict]
        scores = np.array(scores).reshape(len(C_range), len(gamma_range))


        # Make a nice figure
        # plt.figure(figsize=(8, 6))
        # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
        # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
        # plt.xlabel('gamma')
        # plt.ylabel('C')
        # plt.colorbar()
        # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        # plt.yticks(np.arange(len(C_range)), C_range)
        # plt.show()
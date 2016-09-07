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

import os
import re
import random
import pandas as pd
import numpy as np
import itertools
import re



class DataProcess(object):
    """docstring for DataProcess"""
    def __init__(self):
        super(DataProcess, self).__init__()

    def load_data(self, filename):

        print "Read the dataset"
        print '=' * 50
        cols = ['Sentiment','SentimentText', 't', 'Similarity','b1','b2','b3','b4','b5']
        df = pd.read_csv(filename, sep=',', header=None, names=cols)
        data_label = list(df.Sentiment)
        # print data_label

        data_content = list(df.SentimentText)
        data_similarity = list(df.Similarity)
        data_b1 = list(df.b1)
        data_b2 = list(df.b2)
        data_b3 = list(df.b3)
        data_b4 = list(df.b4)
        data_b5 = list(df.b5)

        # data_similarity = list(df.Similarity)
        # print data_content
        # return the data content and their label



        # return data_content, data_label, data_similarity
        return data_content, data_label,data_similarity,data_b1,data_b2,data_b3,data_b4,data_b5

    def extract_n_p_total(self, data_label):
        n=p=0
        for i in range(len(data_label)):
            if int(data_label[i]) == 0:
                n=n+1
            else:
                p=p+1

        print "positive tweet: ", p, "of ",len(data_label),"\n"
        print "negative  tweet: ", n, "of ",len(data_label),"\n"


    def pre_process(self, data):
        print "Pre-process the data"
        print '=' * 50
        filtered_data = []
        for temp in data:
            # to lowercase
            temp = temp.lower()
            # remove the &quot
            temp = re.sub(r'^&quot;', '', temp, flags=re.MULTILINE)
            # remove digits
            temp = ''.join(i for i in temp if not i.isdigit())
            # temp = ['*' if all(c.isdigit() is False for c in x) is False else x for x in temp]

            words = ""
            temp_list = temp.split()
            # remove http/www/#/@
            for i in temp_list:
                if i.find("http") == -1 and i.find("www") == -1 and i.find("#") == -1 and i.find("@") == -1:
                    # handle the multiplication
                    word = ''.join(ch for ch, _ in itertools.groupby(i))
                    words = words + " " + word
            filtered_data.append(words)
        # print filtered_data
        return filtered_data

    def lemmatizer(self, data):
        filtered_data = []
        lmtzr = LancasterStemmer()
        for item in data:
            word_list = re.sub("[^\w]", " ", item).split()
            #word_list = item.split()
            words = ""
            for word in word_list:
                words = words + " " + lmtzr.stem(word)
            filtered_data.append(words)
        return filtered_data


    # remove the digital number
    def removeDigits(self, string):
        results = ''.join(i for i in string if not i.isdigit())
        return results

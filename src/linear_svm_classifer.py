import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
import codecs
from os.path import join, dirname, abspath
from sklearn.metrics import accuracy_score

#Author: Atri Mandal

#Usage: python ml_classifier_pipeline.py train_50k.csv test_10k.csv svmtest

#reload(sys)
#sys.setdefaultencoding('utf8')
csv.field_size_limit(sys.maxsize)

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
resultsFile = sys.argv[3]

training_ids = train.iloc[:,0]
train_X = train.iloc[:,7].astype('U').values	#clean_text
train_Y = train.iloc[:,8].astype('U').values	#assigned_group
names=["LinearSVM"]
classifiers = [LinearSVC(random_state=0)]

j=0
for name, classifier in zip(names, classifiers):
    j=j+1

    print('CLASSIFIER ' + str(j) + ' BEGIN : ' + name)

    t=time()

    
    clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.8, min_df=5)),('tfidf', TfidfTransformer()),('clf', classifier),])
    print  (clf.get_params(deep=False))

    clf.fit(train_X,train_Y)
    duration = time() - t
    print ('Trained the classifier. Took ' + str(duration) + ' seconds...')

    test_ids = test.iloc[:,0]
    test_X = test.iloc[:,7].astype('U').values
    test_Y = test.iloc[:,8].astype('U').values

    proba = clf.decision_function(test_X)
#     platt_proba = (1./(1.+np.exp(-proba)))
    platt_proba = np.exp(proba) / np.sum(np.exp(proba), axis=1, keepdims=True)
    predictions = clf.predict(test_X)
    

    opfname=name + '-' + resultsFile + '.results.csv'
    opf = codecs.open(opfname, "w", "utf-8")
    opf.write('Ticketid,Actual,Predicted,Hit/Miss\n')

    size = len(test_Y)
    print ('Total size of test dataset: ' + str(size))
    for i in range(size):
            ticketid = test_ids[i]
            platt_predict = str(platt_proba[i].max())
            if test_Y[i] == predictions[i]:
                    opf.write(str(ticketid) + ',' + str(test_Y[i]) + ',' + str(predictions[i]) + ',' + str(platt_predict) + ',Hit\n')
            else:
                    opf.write(str(ticketid) + ',' + str(test_Y[i]) + ',' + str(predictions[i]) + ',' + str(platt_predict) + ',Miss\n')
    
    opf.close()

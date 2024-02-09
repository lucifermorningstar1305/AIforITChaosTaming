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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
import codecs
from os.path import join, dirname, abspath
from sklearn.metrics import accuracy_score

#Author: Atri Mandal

#Usage: python ml_classifier_pipeline_predict.py train_10k.csv test_10k.csv mltest
#Runs all the ML classifiers in one shot

#reload(sys)
#sys.setdefaultencoding('utf8')
csv.field_size_limit(sys.maxsize)

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
resultsFile = sys.argv[3]

training_ids = train.iloc[:,0]
train_X = train.iloc[:,7]
train_Y = train.iloc[:,8]

test_ids = test.iloc[:,0]
test_X = test.iloc[:,7]
test_Y = test.iloc[:,8]


###version with no tf-idf or stopword
#clf = Pipeline([('vect', CountVectorizer()),('clf', SGDClassifier()),])
###version with no stopword removal
#clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier()),])
###version with stopword removal
#clf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf', SGDClassifier()),])
###version with tfidf and ngrams
#clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer()),('clf', LinearSVC(loss='hinge', penalty='l2')),])
#clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer()),('clf', SVC(probability=True)),])
#non linear SVM with RBF kernel
#clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer()),('clf', SVC(kernel='rbf')),])
#Version with multinomial naive bayes
#clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
#version with random forest clasifier
#clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(n_estimators = 100)),])
###version with tfidf and ngrams
#clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer()),('clf', SGDClassifier()),])
#clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer()),('clf', SVC(kernel='linear')),])
#names=["LR"]
#classifiers = [LogisticRegression()]

#names=["MultinomialNB","RandomForest","AdaBoost","LogisticRegression","GradientBoosting"]
#classifiers = [MultinomialNB(fit_prior=False),RandomForestClassifier(n_estimators=100),AdaBoostClassifier(n_estimators=100),LogisticRegression(),GradientBoostingClassifier(random_state=0)]
names=["KNeighborsClassifier"]
classifiers = [KNeighborsClassifier(n_neighbors=7)]

j=0
for name, classifier in zip(names, classifiers):
    j=j+1

    print ('CLASSIFIER ' + str(j) + ' BEGIN : ' + name)
    sys.stdout.flush()

    t=time()

    clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),max_df=0.8,min_df=5)),('tfidf', TfidfTransformer()),('clf', classifier),])
    print  (clf.get_params(deep=False))

    clf.fit(train_X,train_Y)
    duration = time() - t
    print ('Trained the classifier. Took ' + str(duration) + ' seconds...')
    sys.stdout.flush()

    predictions = clf.predict(test_X)
    accScore=accuracy_score(test_Y, predictions)
    print ('Accuracy: ', str(accScore)) 
    sys.stdout.flush()

    probabilities = clf.predict_proba(test_X)

    opfname=name + '-' + resultsFile + '.results.csv'
    opf = codecs.open(opfname, "w", "utf-8")
    opf.write('Ticketid,Actual,Predicted,Hit/Miss\n')

    size = len(test_Y)
    for i in range(size):
            ticketid = test_ids[i]
            if test_Y[i] == predictions[i]:
                    opf.write(ticketid + ',' + test_Y[i] + ',' + predictions[i] + ',' + str(np.max(probabilities[i])) + ',Hit\n')
            else:
                    opf.write(ticketid + ',' + test_Y[i] + ',' + predictions[i] + ',' + str(np.max(probabilities[i])) + ',Miss\n')
    
    opf.close()

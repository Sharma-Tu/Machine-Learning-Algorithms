# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:01:35 2019

@author: Tushar
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score

#X = pd.read_csv('./data/bank_X.csv', header = None)
#y = pd.read_csv('./data/bank_y.csv', header = None)

X = pd.read_csv('mushroom_X.csv', header = None)
y = pd.read_csv('mushroom_y.csv', header = None)

print(X.shape,y.shape)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, np.asarray(y)[0], train_size = 
                                                    0.8, test_size = 0.2,
                                                   random_state=1)

print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


neural = MLPClassifier(hidden_layer_sizes=(60, ), 
              activation='logistic', 
              solver='sgd', 
              alpha=0.0001, 
              batch_size='auto', 
              learning_rate='constant', 
              learning_rate_init=0.001, 
              max_iter=2000,
              random_state=None, 
              tol=0.0001, 
              early_stopping=False)


neural.fit(X_train,y_train)

score = neural.score(X_test, y_test)

fpr, tpr, thr = roc_curve(y_test, neural.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

print("score: ",score, 
      "\nauc: ",roc_auc, 
      "\niterations: " , neural.n_iter_, 
      "\nlayers: ", neural.n_layers_)

fpr, tpr, thr = roc_curve(y_train, neural.predict_proba(X_train)[:,1])
roc_auc = auc(fpr, tpr)

score = neural.score(X_train, y_train)
print("score: ",score, 
      "\nauc: ",roc_auc, 
      "\niterations: " , neural.n_iter_, 
      "\nlayers: ", neural.n_layers_)

from timeit import default_timer as timer
from sklearn.model_selection import cross_val_score
#digits = load_digits()

start = timer()
neurons = 2 ** np.asarray([i for i in range(0,10)])
neural_list = [MLPClassifier(hidden_layer_sizes=(neurons_, ), 
              activation='logistic', 
              solver='sgd', 
              alpha=0.0001, 
              batch_size='auto', 
              learning_rate='constant', 
              learning_rate_init=0.001, 
              max_iter=2000,
              random_state=None, 
              tol=0.0001, 
              early_stopping=False) for neurons_ in neurons]

#grid = GridSearchCV(neural,  {'hidden_layer_sizes': neurons}, n_jobs = -1, cv=None)
#grid.fit(X_train, y_train)
neural_fit = [i.fit(X_train, y_train) for i in neural_list]
score = [i.score(X_train, y_train) for i in neural_fit]

accuracy = [cross_val_score(i, X_train, y_train, cv=5).mean() for i in neural_fit]

fpr = []
tpr = []
thr = []
for i in neural_fit:
    fpr.append(roc_curve(y_train, i.predict_proba(X_train)[:,1])[0]), 
    tpr.append(roc_curve(y_train, i.predict_proba(X_train)[:,1])[1]), 
    thr.append(roc_curve(y_train, i.predict_proba(X_train)[:,1])[2])

roc_auc = [auc(i, j) for i,j in zip(fpr, tpr)]

gridTime = timer()-start
print(gridTime, score, accuracy)

score_test = [i.score(X_test, y_test) for i in neural_fit]
fpr = []
tpr = []
thr = []
for i in neural_fit:
    fpr.append(roc_curve(y_test, i.predict_proba(X_test)[:,1])[0]), 
    tpr.append(roc_curve(y_test, i.predict_proba(X_test)[:,1])[1]), 
    thr.append(roc_curve(y_test, i.predict_proba(X_test)[:,1])[2])
roc_auc_test = [auc(i, j) for i,j in zip(fpr, tpr)]
accuracy_test = [accuracy_score(y_test, i.predict(X_test)) for i in neural_fit]

import matplotlib.pyplot as plt
plt.semilogx(neurons, accuracy, neurons, accuracy_test, basex =2)
plt.xlabel('neurons')
plt.ylabel('accuracy')
plt.title('Neural Network Performance')
#print(grid.best_params_)
#print('accuracy =', grid.best_score_)

from timeit import default_timer as timer
from sklearn.model_selection import cross_val_score
#digits = load_digits()

start = timer()
max_iter_list = [i for i in range(100, 2000, 200)]
neural_list = [MLPClassifier(hidden_layer_sizes=4, 
              activation='logistic', 
              solver='sgd', 
              alpha=0.0001, 
              batch_size='auto', 
              learning_rate='constant', 
              learning_rate_init=0.001, 
              max_iter=max_iter_,
              random_state=None, 
              tol=0.0001, 
              early_stopping=False) for max_iter_ in max_iter_list]

#grid = GridSearchCV(neural,  {'hidden_layer_sizes': neurons}, n_jobs = -1, cv=None)
#grid.fit(X_train, y_train)
neural_fit = [i.fit(X_train, y_train) for i in neural_list]
score = [i.score(X_train, y_train) for i in neural_fit]

accuracy = [cross_val_score(i, X_train, y_train, cv=5).mean() for i in neural_fit]

fpr = []
tpr = []
thr = []
for i in neural_fit:
    fpr.append(roc_curve(y_train, i.predict_proba(X_train)[:,1])[0]), 
    tpr.append(roc_curve(y_train, i.predict_proba(X_train)[:,1])[1]), 
    thr.append(roc_curve(y_train, i.predict_proba(X_train)[:,1])[2])

roc_auc = [auc(i, j) for i,j in zip(fpr, tpr)]

gridTime = timer()-start
print(gridTime, score, accuracy)

score_test = [i.score(X_test, y_test) for i in neural_fit]
fpr = []
tpr = []
thr = []
for i in neural_fit:
    fpr.append(roc_curve(y_test, i.predict_proba(X_test)[:,1])[0]), 
    tpr.append(roc_curve(y_test, i.predict_proba(X_test)[:,1])[1]), 
    thr.append(roc_curve(y_test, i.predict_proba(X_test)[:,1])[2])
roc_auc_test = [auc(i, j) for i,j in zip(fpr, tpr)]
accuracy_test = [accuracy_score(y_test, i.predict(X_test)) for i in neural_fit]

import matplotlib.pyplot as plt
plt.plot(max_iter_list, accuracy, max_iter_list, accuracy_test)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('Neural Network Performance')


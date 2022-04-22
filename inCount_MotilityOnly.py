#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:07:15 2022

@author: kesaprm
"""

from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# iris = datasets.load_iris()

M_data = pd.read_pickle("Ms_for_SVM_incCount.txt") 

# X = iris.data[:, :2]t
# Y = iris.target
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.80, test_size=0.20, random_state=101)


X_data = M_data.iloc[:,4:].to_numpy()
Y_data = M_data.iloc[:,0].to_numpy()


##Shuffle and split data
import sklearn.model_selection as model_selection
from sklearn.utils import shuffle


all_data, all_labels = shuffle(X_data, Y_data, random_state=42)

X_train, X_test, y_train, y_test = model_selection.train_test_split(all_data, all_labels , train_size=0.80, test_size=0.20, random_state=101)




#deal imbalanced data
import imblearn
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


ros = RandomOverSampler(random_state=42)

x_ros, y_ros = ros.fit_resample(X_train, y_train)

print('Original dataset shape', Counter(y_train))
print('Resample dataset shape', Counter(y_ros))


from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X_train, y_train)

print('Original dataset shape', Counter(y_train))
print('Resample dataset shape', Counter(y_ros))


###SVM classifier
rbf = svm.SVC(kernel='sigmoid', gamma=0.9, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, gamma=0.9).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))


rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))



#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(poly_pred, y_test, normalize='true')
print(cm)
sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='.0%',xticklabels=['M0','M1','M2'], yticklabels=['M0','M1','M2'])



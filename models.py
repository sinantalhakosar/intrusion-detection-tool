import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import time
import multiprocessing 
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import plot_table

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import random

# LSTM for sequence classification in the IMDB dataset


# fix random seed for reproducibility
np.random.seed(1234)

import dataset_operations as dbo

colNames = list()

row_labels = []
table_vals = []

def fs_percentile(X_train, Y_train, X_test):
    global colNames
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler1.transform(X_train) 
    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test=scaler2.transform(X_test) 
    np.seterr(divide='ignore', invalid='ignore');
    selector=SelectPercentile(f_classif, percentile=10)
    X_new = selector.fit_transform(X_train,Y_train)
    true=selector.get_support()
    newcolindex=[i for i, x in enumerate(true) if x]
    newcolname=list(colNames[i] for i in newcolindex)
    return X_new,X_test[:,newcolindex],newcolname, newcolindex

def feature_selection(index, dataset):
    global colNames
    df = dbo.get_dataframe(index)
    #df = dbo.concat_dataframes()
    df = dbo.label_replace(df)
    df = df.astype('float64')
    train_df, test_df = dbo.dataset_split_and_label_replace(df,index)
    X_train = train_df.drop(' Label',1)
    Y_train = train_df.loc[:,[" Label"]]
    X_test = test_df.drop(' Label',1)
    Y_test = test_df.loc[:,[" Label"]]
    colNames=list(X_train)
    X_train, Y_train = dbo.clean_dataset(X_train, Y_train)
    X_test, Y_test = dbo.clean_dataset(X_test, Y_test)
    X_new, X_test2, percentile_features, newcolindex = fs_percentile(X_train,Y_train.values.ravel(),X_test)
    return X_new, Y_train, X_test2, Y_test

def decisionTreeModel(X_train,Y_train,X_test,Y_test,index):
    clf=DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    scorer(Y_test, Y_pred,'DecisionTreeClassifier',index)
    plotter(clf,X_test,Y_test)


def adaBoostModel(X_train,Y_train,X_test,Y_test,index):
    classifier = AdaBoostClassifier(n_estimators=500,learning_rate=1)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    scorer(Y_test, Y_pred,'AdaBoostClassifier',index)

def randomForestModel(X_train,Y_train,X_test,Y_test,index):
    clf=RandomForestClassifier(n_estimators=100)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,Y_train.values.ravel())
    Y_pred=clf.predict(X_test)
    scorer(Y_test, Y_pred,'RandomForestClassifier',index)
    plotter(clf,X_test,Y_test)

def lstm(X_train,Y_train,X_test,Y_test,index):
    
    #Create a Multi-Layer Perceptron
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test);
    scorer(Y_test, Y_pred,'MLP',index)

    #Create a Gaussian Naive Bayes Classifier
    # gnb = GaussianNB()
    # gnb.fit(X_train,Y_train)
    # Y_pred=gnb.predict(X_test)
    # scorer(Y_test, Y_pred,'NB',index)

def scorer(Y_test, Y_pred,modelname,index):
    print("####### ",modelname, " #######")
    print(confusion_matrix(Y_test, Y_pred))
    print("Acc: ", metrics.accuracy_score(Y_test,Y_pred))
    print('Precision: ', metrics.precision_score(Y_test,Y_pred,average='weighted'))
    print('Recall: ', metrics.recall_score(Y_test,Y_pred,average='weighted'))
    print('F1: ', metrics.f1_score(Y_test,Y_pred,average='weighted'))

    row_labels.append(str(index))
    table_vals.append([str(metrics.accuracy_score(Y_test,Y_pred))[0:5],
                    str(metrics.precision_score(Y_test,Y_pred,average='weighted'))[0:5],
                    str(metrics.recall_score(Y_test,Y_pred,average='weighted'))[0:5],
                    str(metrics.f1_score(Y_test,Y_pred,average='weighted'))[0:5]])

def plotter(clf,X_test,Y_test):
    rfecv = RFECV(estimator=clf, step=1, cv=10, scoring='accuracy')
    rfecv.fit(X_test, Y_test)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.title('RFECV')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

def runner(index, dataset):
    if "Monday" in dataset:
        return
    print(dataset)
    X_train, Y_train, X_test, Y_test = feature_selection(index,dataset)
    #adaBoostModel(X_train,Y_train,X_test,Y_test,index)
    #decisionTreeModel(X_train,Y_train,X_test,Y_test,index)
    #randomForestModel(X_train,Y_train,X_test,Y_test,index)
    lstm(X_train,Y_train,X_test,Y_test,index)

# no multiprocessing, !!for drawing table!!, otherwise no shared memory, overrides
# for a in range(8):
#     if( a == 3 ): # All benign, no feature selection, Monday-WorkingHours
#         continue
#     print(a, "->" ,dbo.find_all_datasets()[a])
#     feature_selection(a,dbo.find_all_datasets()[a])

# plot_table.draw(row_labels,table_vals)
    
a=5
# for i, dataset in enumerate(dbo.find_all_datasets()):
#     print(dbo.find_all_datasets()[a])
#     process = multiprocessing.Process(target=runner, args=(a,dbo.find_all_datasets()[a], ))
#     process.start()
#     process.join()
#     break

runner(a,dbo.find_all_datasets()[a])
    
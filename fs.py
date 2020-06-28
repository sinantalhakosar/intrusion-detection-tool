import pandas as pd
import numpy as np
import sys
import time
from blessings import Terminal
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
import random
random.seed(1234)
import dataset_operations as dbo
import plot_table

colNames = list()
term = Terminal()

row_labels = []
table_vals = []

def fs_rfe(X_train, Y_train, colnames):
    clf = DecisionTreeClassifier(random_state=0)
    #rank all features, i.e continue the elimination until the last one
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X_train, Y_train)
    print ("Features sorted by their rank:")
    print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames)))

def fs_percentile(X_train, Y_train, X_test):
    global colNames
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler1.transform(X_train) 
    scaler5 = preprocessing.StandardScaler().fit(X_test)
    X_test=scaler5.transform(X_test) 
    np.seterr(divide='ignore', invalid='ignore');
    selector=SelectPercentile(f_classif, percentile=10)
    X_new = selector.fit_transform(X_train,Y_train)
    true=selector.get_support()
    newcolindex=[i for i, x in enumerate(true) if x]
    newcolname=list(colNames[i] for i in newcolindex)
    #print("newcolindex -> ",newcolindex )
    return X_new,X_test,newcolname, newcolindex

def fs_extra_trees_classifier(X_train, Y_train, importance_count):
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X_train, Y_train)
    sorted_l = np.sort(model.feature_importances_)
    reversed_arr = sorted_l[::-1]
    ind = []
    for sort in reversed_arr[:importance_count]:
        result = np.where(model.feature_importances_ == sort)
        ind.append(result[0][0])
    return [X_train.columns[i] for i in ind]
    #return np.sum(model.feature_importances_)

def feature_selection(index, dataset):
    global colNames,row_labels,table_vals
    df = dbo.get_dataframe(index)
    # with term.location(y=index*2+1):
    #     sys.stdout.write("\033[K")
    print("loading dataset:",index, "-> 100%")
    time.sleep(1)
    df = dbo.dataset_normalizer(df,index)
    #with term.location(y=index*2+1):
        #sys.stdout.write("\033[K")
    print("normalizing dataset:",index,"-> 100%")
    time.sleep(1)
    train_df, test_df = dbo.dataset_split_and_label_replace(df,index)
    #with term.location(y=index*2+1):
    #    sys.stdout.write("\033[K")
    print("splitting dataset:",index,"-> 100%")
    time.sleep(1)
# train_df =  dbo.label_replace(train_df)
# test_df = dbo.label_replace(test_df)
#df = dbo.concat_dataframes()
    # print(train_df[" Label"].head())
    # print("//////////////")
    # print(test_df[" Label"].head())

    #with term.location(y=index*2+1):
    #    sys.stdout.write("\033[K")
    print("feature selecting for dataset:",index,"-> 0%")
    
    X_train = train_df.drop(' Label',1)
    Y_train = train_df.loc[:,[" Label"]]
    X_test = test_df.drop(' Label',1)
    Y_test = test_df.loc[:,[" Label"]]
    colNames=list(X_train)

    X_train, Y_train = dbo.clean_dataset(X_train, Y_train)
    X_test, Y_test = dbo.clean_dataset(X_test, Y_test)

    X_new, X_test, percentile_features, newcolindex = fs_percentile(X_train,Y_train.values.ravel(),X_test)
    X_test2=X_test[:,newcolindex]

    
    #fs_rfe(X_new,Y_train,percentile_features) #another feature selection method
    #extra_trees_classifier_features = fs_extra_trees_classifier(X_train, Y_train,len(percentile_features))
    # with term.location(y=index+1):
    #     sys.stdout.write("\033[K")
    #print("selected features for dataset:",index, "-> ",percentile_features)
    #print(percentile_features)
    #print(extra_trees_classifier_features)
    #print("------------")

    #################                  DECISION TREE                           ###################
    # clf_rfeDoS=DecisionTreeClassifier(random_state=0)
    # clf_rfeDoS.fit(X_new, Y_train)
    # Y_pred = clf_rfeDoS.predict(X_test2)
    # print(confusion_matrix(Y_test, Y_pred))
    # print("Accuracy: ", metrics.accuracy_score(Y_test,Y_pred))
    # print("Precision: ", metrics.precision_score(Y_test,Y_pred,average='weighted'))
    # print("Recall: ", metrics.recall_score(Y_test,Y_pred,average='weighted'))
    # print("F1 score: ", metrics.f1_score(Y_test,Y_pred,average='weighted'))


    # row_labels.append(str(index))
    # table_vals.append([str(metrics.accuracy_score(Y_test,Y_pred))[0:5],
    #                 str(metrics.precision_score(Y_test,Y_pred,average='weighted'))[0:5],
    #                 str(metrics.recall_score(Y_test,Y_pred,average='weighted'))[0:5],
    #                 str(metrics.f1_score(Y_test,Y_pred,average='weighted'))[0:5]])



    ##############                       ADABOOST                       ######################
    classifier = AdaBoostClassifier(n_estimators=500,learning_rate=1)
    classifier.fit(X_new, Y_train)
    Y_pred = classifier.predict(X_test2)
    print(confusion_matrix(Y_test, Y_pred))
    print("Accuracy: ", metrics.accuracy_score(Y_test,Y_pred))
    print("Precision: ", metrics.precision_score(Y_test,Y_pred,average='weighted'))
    print("Recall: ", metrics.recall_score(Y_test,Y_pred,average='weighted'))
    print("F1 score: ", metrics.f1_score(Y_test,Y_pred,average='weighted'))


    row_labels.append(str(index))
    table_vals.append([str(metrics.accuracy_score(Y_test,Y_pred))[0:5],
                    str(metrics.precision_score(Y_test,Y_pred,average='weighted'))[0:5],
                    str(metrics.recall_score(Y_test,Y_pred,average='weighted'))[0:5],
                    str(metrics.f1_score(Y_test,Y_pred,average='weighted'))[0:5]])

    ####################################


# no multiprocessing, !!for drawing table!!, otherwise no shared memory, overrides
for a in range(8):
    if( a == 3 ): # All benign, no feature selection, Monday-WorkingHours
        continue
    for i, dataset in enumerate(dbo.find_all_datasets()):
        print(a, "->" ,dbo.find_all_datasets()[a])
        feature_selection(a,dbo.find_all_datasets()[a])
        break

plot_table.draw(row_labels,table_vals)


# multiprocess
# for a in range(8):
#     print(a)
#     for i, dataset in enumerate(dbo.find_all_datasets()):
#         print(a, "->" ,dbo.find_all_datasets()[a])
#         process = multiprocessing.Process(target=feature_selection, args=(a,dbo.find_all_datasets()[a], ))
#         process.start()
#         process.join()
#         break


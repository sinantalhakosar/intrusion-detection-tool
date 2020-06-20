import pandas as pd
import numpy as np
import sys
import multiprocessing 
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import ExtraTreesClassifier

import dataset_operations as dbo

colNames = list()


def fs_percentile(X_train, Y_train):
    global colNames
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler1.transform(X_train) 
    np.seterr(divide='ignore', invalid='ignore');
    selector=SelectPercentile(f_classif, percentile=10)
    selector.fit_transform(X_train,Y_train)
    true=selector.get_support()
    newcolindex=[i for i, x in enumerate(true) if x]
    newcolname=list(colNames[i] for i in newcolindex)
    #print(newcolname)
    return newcolname

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
    global colNames
    df = dbo.get_dataframe(index)
    df = dbo.dataset_normalizer(df)
    train_df, test_df = dbo.dataset_split_and_label_replace(df)
# train_df =  dbo.label_replace(train_df)
# test_df = dbo.label_replace(test_df)
#df = dbo.concat_dataframes()
    # print(train_df[" Label"].head())
    # print("//////////////")
    # print(test_df[" Label"].head())

    X_train = train_df.drop(' Label',1)
    Y_train = train_df[" Label"]
    X_test = test_df.drop(' Label',1)
    Y_test = test_df[" Label"]
    colNames=list(X_train)

    X_train, Y_train = dbo.clean_dataset(X_train, Y_train)
    X_test, Y_test = dbo.clean_dataset(X_test, Y_test)

    percentile_features = fs_percentile(X_train,Y_train)
    print(dataset)
    print(percentile_features)
    extra_trees_classifier_features = fs_extra_trees_classifier(X_train, Y_train,len(percentile_features))
    print(extra_trees_classifier_features)

    print("------------")

#gridsearchcv kullan, false positive table ı kullan, r-1?


for i, dataset in enumerate(dbo.find_all_datasets()):
    process = multiprocessing.Process(target=feature_selection, args=(i,dataset, ))
    process.start()
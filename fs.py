import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pd.read_csv("friday_botnet.csv",low_memory=False)
df = df.loc[:, ~df.columns.str.replace("(\.\d+)$", "").duplicated()]
#df_test = pd.read_csv("friday_botnet.csv", header=None, names = col_names)

categorical_columns = list()
for col_name in df.columns:
    if df[col_name].dtypes == 'object' and col_name != " Label":
        for i in range(0,len(df[col_name].values)):
            try:
                df[col_name].values[i] = float(df[col_name].values[i])
            except:
                #print(type(df[col_name].values[i]))
                categorical_columns.append(col_name)
                break
        #print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


 # Get the categorical values into a 2D numpy array
if len(categorical_columns) > 0:
    df_categorical_values = df[categorical_columns]

    unique_protocol=sorted(df[' Label'].unique())

    string1 = 'Label_'
    unique_protocol2=[string1 + x for x in unique_protocol]


    df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

    enc = OneHotEncoder(categories="auto")
    print("---------")
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    print("+++++++++")
    df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=unique_protocol2)

    print(df_cat_data.head())

    newdf=df.join(df_cat_data)
    for col_name in categorical_columns:
        newdf.drop(col_name, axis=1, inplace=True)

    print(newdf.shape)
# labeldf = df[" Label"]
# newlabeldf=labeldf.replace({"BENIGN":2,"Bot":3})
# df[" Label"] = newlabeldf
# print(df[" Label"].head())
# Bot_df=df[df[' Label'].isin([2,3])]
# print('Dimensions of Bot_df:' ,Bot_df.shape)

X_Bot = df.drop(' Label',1)
Y_Bot = df[" Label"]
colNames=list(X_Bot)

def clean_dataset(df,ydf):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True,axis=1, how='all')
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64), ydf[indices_to_keep]


X_Bot, Y_Bot = clean_dataset(X_Bot, Y_Bot)
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X_Bot, Y_Bot)
print(model.feature_importances_)

print(np.sum(model.feature_importances_))



# from sklearn import preprocessing
# print("++",X_Bot.shape)

# #Y_Bot = clean_dataset(Y_Bot)
# scaler1 = preprocessing.StandardScaler().fit(X_Bot)
# X_Bot=scaler1.transform(X_Bot) 

# print(X_Bot.shape, "--",Y_Bot.shape)
# print(Y_Bot)
# from sklearn.feature_selection import SelectPercentile, f_classif
# np.seterr(divide='ignore', invalid='ignore');
# selector=SelectPercentile(f_classif)
# print("----")
# print(selector)
# X_newDoS = selector.fit_transform(X_Bot,Y_Bot)
# print(X_newDoS)

# true=selector.get_support()
# print("true:",true)
# newcolindex_DoS=[i for i, x in enumerate(true) if x]
# print("a:",newcolindex_DoS)
# newcolname_DoS=list( colNames[i] for i in newcolindex_DoS )
# print(newcolname_DoS)

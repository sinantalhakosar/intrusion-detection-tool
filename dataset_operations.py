from os import listdir
from blessings import Terminal
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import random
random.seed(1234)


path_to_dir = "../"
term = Terminal()

def dataset_count():
    return len(find_all_datasets(path_to_dir))

def find_all_datasets(path_to_dir=path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

csvpaths = find_all_datasets(path_to_dir)

def get_dataframe(dataset_number=0):
    #print(csvpaths[dataset_number])
    #with term.location(y=dataset_number*2+1):
    print("loading dataset:",dataset_number, "-> 0%")
    df = pd.read_csv(path_to_dir+csvpaths[dataset_number],low_memory=False)
    df = df.loc[:, ~df.columns.str.replace("(\.\d+)$", "").duplicated()]
    return df

def concat_dataframes(csvpaths=csvpaths):
    df_list = list()
    for i,path in enumerate(csvpaths):
        print ("\r Loading... {}".format(i*100./len(csvpaths))+str("%"), end="")
        df_sing = get_dataframe(i)
        df_list.append(df_sing)
    df = pd.concat(df_list)
    return df

def dataset_normalizer(df, index):
    categorical_columns = list()
    label_list = list()
    #with term.location(y=index*2+1):
        #sys.stdout.write("\033[K")
    print("normalizing dataset:",index,"-> 0%")
    for col_name in df.columns:
        if df[col_name].dtypes == 'object' and col_name != " Label":
                for i in range(0,len(df[col_name].values)):
                    try:
                        df[col_name].values[i] = float(df[col_name].values[i])
                    except:
                        #print(type(df[col_name].values[i]))
                        categorical_columns.append(col_name)
                        break
    #print(df[" Label"].unique())
    # if len(categorical_columns) > 0:
    #     df_categorical_values = df[categorical_columns]

    #     unique_protocol=sorted(df[' Label'].unique())

    #     string1 = 'Label_'
    #     unique_protocol2=[string1 + x for x in unique_protocol]


    #     df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

    #     enc = OneHotEncoder(categories="auto")
    #     df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    #     df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=unique_protocol2)

    #     #print(df_cat_data.head())

    #     newdf=df.join(df_cat_data)
    #     for col_name in categorical_columns:
    #         newdf.drop(col_name, axis=1, inplace=True)
    #     df = newdf
        
    return df

def label_replace(df):
    for i,unique in enumerate(df.loc[:,[' Label']].drop_duplicates().values):
        df = df.replace({unique[0]:i})
    return df

def dataset_split_and_label_replace(df,index):
    #with term.location(y=index*2+1):
        #sys.stdout.write("\033[K")
    print("splitting dataset:",index,"-> 0%")
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df = label_replace(train_df)
    test_df = label_replace(test_df)
    return train_df, test_df

def clean_dataset(df,ydf):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True,axis=1, how='all')
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64), ydf[indices_to_keep]

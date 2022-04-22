import time
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import re
import requests
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

set_config(display='diagram') # Useful for display the pipeline



def data_inspection(DATA_PATH):
    
    df     = pd.read_csv(DATA_PATH + "train.csv", index_col='PassengerId')
    df_test = pd.read_csv(DATA_PATH + "test.csv",  index_col='PassengerId')
    print("\n","Train DataFrame:", df.shape)
    print("Test DataFrame: ", df_test.shape,"\n")
    print("Train Dataset info: ")
    print(df.info(),"\n")
    print("Test Dataset info: ")
    print(df_test.info())
    

def data_cleaning(DATA_PATH):
    df     = pd.read_csv(DATA_PATH + "train.csv", index_col='PassengerId')
    df_test = pd.read_csv(DATA_PATH + "test.csv",  index_col='PassengerId')
    df['Title'] = df['Name'].apply(lambda i: (i.split(',')[1].split('.')[0]).lstrip())
    df_test['Title'] = df_test['Name'].apply(lambda i: (i.split(',')[1].split('.')[0]).lstrip())
    title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"}

    df["Title"] = df["Title"].map(title_dictionary)
    df_test["Title"] = df_test["Title"].map(title_dictionary)

    return df,df_test

   


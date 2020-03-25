# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:33:02 2020

@author: Jerron
"""

import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sys import argv
from joblib import dump

######################### USER DEFINED FUNCTIONS ##############################

#part_day function transform time variable into part of day categorical variable    
def part_day(series):
    if (series >= 6) & (series <= 11):
        return "morning"
    if (series >= 12) & (series <= 17):
        return "afternoon"
    if (series >= 18) & (series <= 23):
        return "evening"
    if (series >= 0) & (series <= 5):
        return "night"
  
#seasons function transform month variable into season categorical variable    
def seasons(series):
    if (series == 12) | (series == 1) | (series ==2):
        return "winter"
    if (series >= 3) & (series <= 5):
        return "spring"
    if (series >= 6) & (series <= 8):
        return "summer"
    if (series >= 9) & (series <= 11):
        return "autumn"

#df_xform_col performs one-hot encoding for a categorical variable
def df_xform_col(df,col_name):
    dummy = pd.get_dummies(df[[col_name]],drop_first=True)
    df = pd.concat([df,dummy],axis=1)
    dummy_var = dummy.columns.tolist()
    return [df,dummy_var]
    #returns transformed dataframe "df" and dummy variable column names "dummyVar"

########################### INITIALIZING INPUTS ##############################

if __name__ == '__main__':
    ##default inputs
    data_source = "https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv"
    cat_label = "weather,season,part_of_day"
    cont_bin_label = "temperature,relative-humidity,weekday"
    tar_label = "cbrt_guest-users,cbrt_registered-users"
    test_sz = 0.25
    rand_s_split = 1

else:
    #argv[1] should store the config.json file path
    with open(argv[1],'r') as file:
        pp_dict = (json.loads(file.read()))['preprocessing_dict']
    with open(argv[1],'r') as file:    
        data_source = (json.loads(file.read())['data_source'])

    ##import inputs from config file
    cat_label = pp_dict['cat_label']
    cont_bin_label = pp_dict['cont_bin_label']
    tar_label = pp_dict['tar_label']
    test_sz = pp_dict['test_sz']
    rand_s_split = pp_dict['rand_s_split']
    

############################ APPLYING EDA RESULTS ############################

print("______________________________________________________________________")   
print()
print("Performing data preprocessing...")

#import the data
df = pd.read_csv(data_source)
df["date"] = pd.to_datetime(df["date"])

#apply the data cleansing, enriching, and transforms identified in the EDA phase

##data cleansing:
neg_guest = df["guest-users"]<0 
neg_regis = df["registered-users"]<0
rel_humid_0 = df["relative-humidity"]==0 
index_to_drop = df[rel_humid_0 | neg_guest | neg_regis].index
df = df.drop(index_to_drop,axis = 0)

df["weather"] = np.where(df["weather"]=='loudy','cloudy',df["weather"])
df["weather"] = np.where(df["weather"]=='lear','clear',df["weather"])
df["weather"] = df["weather"].str.lower()

##data enriching:
df["day_week"] = df["date"].dt.dayofweek
df["weekday"] = np.where((df["day_week"]==5)|(df["day_week"]==6),0,1)
df["part_of_day"] = df["hr"].apply(part_day)
df["month"] = df["date"].dt.month
df["season"] = df["month"].apply(seasons)
df["weather"] = np.where(df["weather"]=="heavy snow/rain","snow/rain",df["weather"])
df["weather"] = np.where(df["weather"]=="light snow/rain","snow/rain",df["weather"])

##data transforms:
df["cbrt_guest-users"] = np.cbrt(df["guest-users"])
df["cbrt_registered-users"] = np.cbrt(df["registered-users"])
df["temperature_sq"] = np.square(df["temperature"])
df["relative-humidity_sq"] = np.square(df["relative-humidity"])
df["temperature_cb"] = np.power(df["temperature"],3)
df["relative-humidity_cb"] = np.power(df["relative-humidity"],3)

########################### DATA PREPROCESSING ###############################

#perform one-hot encoding on categorical variables
col_transform = cat_label.split(",")
dummy_var_all = []
dummies_frame = {}
for label in col_transform:
    arr = df_xform_col(df,label)
    df = arr[0]
    dummy_var = arr[1]
    dummy_var_all = np.append(dummy_var_all,dummy_var)
    dummies_frame[label]=dummy_var
#dummy_var_all stores the labels for the dummy variables after performing one-hot encoding of categorical variables
#dummies_frame stores the transformed dummy variable label for use in predicting new instances

print()
print("Saving 'dummies_frame' (required for predicting instances later)...")
with open("mlp/dummies_frame",'wb') as f:
    dump(dummies_frame,f)
print("'dummies_frame' saved.")

#storing the labels for the other variables
other_var = cont_bin_label.split(",")

#storing the labels for target variables
tar_var = tar_label.split(",")

indep_var = np.concatenate((dummy_var_all,other_var),axis = 0)
all_var = np.concatenate((dummy_var_all,other_var,tar_var),axis = 0)

df = df[all_var]

############################ TRAIN-TEST SPLIT ################################

df_train,df_test = train_test_split(df,test_size=test_sz,random_state = rand_s_split)

df_train_x = df_train[indep_var]
df_test_x = df_test[indep_var]

df_train_y = df_train[tar_var]
df_test_y = df_test[tar_var]

print()
print("Data preprocessing completed.")
print()
print("______________________________________________________________________")   
print()
print("Splitting data into training and test sets...")
print()
print("No. of observations in training data: ",df_train_x.shape[0])
print("No. of observations in test data: ",df_test_x.shape[0])
print()
print("Features in training data: ")
for i,col in enumerate(df_train_x.columns):
    print(i+1,col)
print()
print("Target variables: ")
for i,col in enumerate(df_train_y.columns):
    print(i+1,col)
print()

print("______________________________________________________________________")   
print()
print("Saving training data features as 'training_data_features'...")
with open("mlp/training_data_features",'wb') as f:
    dump(df_train_x,f)
print("'training_data_features' saved.")

print()
print("Saving test data features as 'test_data_features'...")
with open("mlp/test_data_features",'wb') as f:
    dump(df_test_x,f)
print("'test_data_features' saved.")

print()
print("Saving training data targets as 'training_data_targets'...")
with open("mlp/training_data_targets",'wb') as f:
    dump(df_train_y,f)
print("'training_data_targets' saved.")

print()
print("Saving test data targets as 'test_data_features'...")
with open("mlp/test_data_targets",'wb') as f:
    dump(df_test_y,f)
print("'test_data_targets' saved.")
print()





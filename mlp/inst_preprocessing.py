# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 00:34:39 2020

@author: Jerron
"""

import numpy as np
import pandas as pd
import json

from sys import argv
from joblib import dump
from joblib import load

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

#df_xform_col performs one-hot encoding for a categorical variable based on dummies from main dataset
def df_xform_col_inst(df,dummies_frame,col_name):
    dummy = pd.get_dummies(df[[col_name]])
    dummy.reindex(columns = dummies_frame[col_name],fill_value=0)
    for col in dummy.columns:
        if col not in dummies_frame[col_name]:
            dummy = dummy.drop(col,axis=1)
    df = pd.concat([df,dummy],axis=1)
    dummy_var = dummy.columns.tolist()
    return [df,dummy_var]
    #returns transformed dataframe "df" and dummy variable column names "dummyVar"

########################### INITIALIZING INPUTS ##############################

if __name__ == '__main__':    
    ##default inputs
    inst_file = "example_inst.csv"
    inst_file = "predictions/"+inst_file
    cat_label = "weather,season,part_of_day"
    cont_bin_label = "temperature,relative-humidity,weekday"

else:    
    with open(argv[1],'r') as file:   
        pp_dict = (json.loads(file.read()))['preprocessing_dict']
        file.close()
    with open(argv[1],'r') as file:   
        inst_file = (json.loads(file.read())['instance_file'])
        inst_file = "predictions/"+inst_file
        file.close()
        
    ##import inputs from config file
    cat_label = pp_dict['cat_label']
    cont_bin_label = pp_dict['cont_bin_label']

############################ APPLYING EDA RESULTS ############################

#import the data
df = pd.read_csv(inst_file)
df["date"] = pd.to_datetime(df["date"])

#apply the data enriching, and transforms identified in the EDA phase

##data enriching:
df["day_week"] = df["date"].dt.dayofweek
df["weekday"] = np.where((df["day_week"]==5)|(df["day_week"]==6),0,1)
df["part_of_day"] = df["hr"].apply(part_day)
df["month"] = df["date"].dt.month
df["season"] = df["month"].apply(seasons)
df["weather"] = np.where(df["weather"]=="heavy snow/rain","snow/rain",df["weather"])
df["weather"] = np.where(df["weather"]=="light snow/rain","snow/rain",df["weather"])

##data transforms:
df["temperature_sq"] = np.square(df["temperature"])
df["relative-humidity_sq"] = np.square(df["relative-humidity"])
df["temperature_cb"] = np.power(df["temperature"],3)
df["relative-humidity_cb"] = np.power(df["relative-humidity"],3)

########################### DATA PREPROCESSING ###############################

dummies_frame = load("mlp/dummies_frame")

#perform one-hot encoding on categorical variables
col_transform = cat_label.split(",")
dummy_var_all = []
for label in col_transform:
    arr = df_xform_col_inst(df,dummies_frame,label)
    df = arr[0]
    dummy_var = arr[1]
    dummy_var_all = np.append(dummy_var_all,dummy_var)
#dummy_var_all stores the labels for the dummy variables after performing one-hot encoding of categorical variables

#storing the labels for the other variables
other_var = cont_bin_label.split(",")

#storing the labels for target variables
# tar_var = tar_label.split(",")

indep_var = np.concatenate((dummy_var_all,other_var),axis = 0)

df = df[indep_var]

print("______________________________________________________________________")
print()
print("Applying transforms to instance(s)...")
print()
print("Saving transformed instance(s) as 'instance_to_predict'...")
with open("mlp/instance_to_predict",'wb') as f:
    dump(df,f)
    f.close()
print("'instance_to_predict' saved.")
print()
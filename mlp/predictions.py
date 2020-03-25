# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:18:56 2020

@author: Jerron
"""

##################### PREDICTIONS ON NEW DATA ################################

import numpy as np
import pandas as pd
import json

from sys import argv
from joblib import load

if __name__ == '__main__':
    ##default inputs
    save_pred = False
    pred_filename = "Predictions.csv"

else:    
    #argv[1] should store the config.json file path
    with open(argv[1],'r') as file:
        pred_dict = (json.loads(file.read()))['prediction_dict']
        file.close()

    save_pred = pred_dict['save_pred']
    pred_filename = pred_dict['pred_filename']

print("______________________________________________________________________")
print()
print("Loading models...")

guest_users_model = load("mlp/cbrt_guest-users_model")
regis_users_model = load("mlp/cbrt_registered-users_model")

print("Models loaded.")

print()
print("Processing predictions...")

inst_data = load("mlp/instance_to_predict")

cbrt_guest_users = guest_users_model.predict(inst_data)
cbrt_regis_users = regis_users_model.predict(inst_data)

guest_users = np.around(np.power(cbrt_guest_users,3))
regis_users = np.around(np.power(cbrt_regis_users,3))

total_users = np.add(guest_users,regis_users)

result = pd.DataFrame()

result["Guest users"] = guest_users.ravel()
result["Registered users"] = regis_users.ravel()
result["Total users"] = total_users.ravel()

print()
print("Predictions:") 
print("(Note: For large number of predictions only the first 5 values are shown)")
print()
print(result.head())

if save_pred == True:
      
    print()
    print("Saving predictions as '"+pred_filename+"'...")
    result.to_csv("predictions/"+pred_filename,index=True)
    print("'"+pred_filename+"' saved.")
    print()














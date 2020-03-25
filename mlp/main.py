# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 00:30:13 2020

@author: Jerron
"""

from sys import argv

if argv[2] == "train":
    import preprocessing
    import mlp
    
elif argv[2] == "predict":
    import inst_preprocessing
    import predictions

# NOTE:
# Script MUST be run in command line with 2 argv variables. Refer to README.
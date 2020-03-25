# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:59:52 2020

@author: Jerron
"""

import numpy as np
import json

from sys import argv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from joblib import dump
from joblib import load

######################### MACHINE LEARNING PIPELINES #########################

# Linear regression pipeline
def linear_pl(df_train_x,df_train_y):
    pipeline_linear=Pipeline([('scaler',StandardScaler()),
                              ('linear_algo',LinearRegression())])

    model = pipeline_linear.fit(df_train_x,df_train_y)
    
    return model

# Lasso regression pipeline
def lasso_pl(df_train_x,df_train_y,hyperparams,rand_s_cv=1):
    pipeline_lasso=Pipeline([('scaler',StandardScaler()),
                             ('lasso_algo',Lasso(random_state=rand_s_cv))])

    grid_param_lasso =  {"lasso_algo__alpha": hyperparams['reg_val']}
    
    if search_type == "random":
        randomsearch_lasso = RandomizedSearchCV(pipeline_lasso,grid_param_lasso,cv=5,n_jobs=-1,n_iter=10,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = randomsearch_lasso.fit(df_train_x,df_train_y.values.ravel())
    
    elif search_type == "grid":
        gridsearch_lasso = GridSearchCV(pipeline_lasso,grid_param_lasso,cv=5,n_jobs=-1,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = gridsearch_lasso.fit(df_train_x,df_train_y)
    
    return best_model


# Ridge regression pipeline
def ridge_pl(df_train_x,df_train_y,hyperparams,rand_s_cv=1):
    pipeline_ridge=Pipeline([('scaler',StandardScaler()),
                             ('ridge_algo',Ridge(random_state=rand_s_cv))])
    
    grid_param_ridge =  {"ridge_algo__alpha": hyperparams['reg_val']}
    
    if search_type == "random":
        randomsearch_ridge = RandomizedSearchCV(pipeline_ridge,grid_param_ridge,cv=5,n_jobs=-1,n_iter=10,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = randomsearch_ridge.fit(df_train_x,df_train_y.values.ravel())
    
    elif search_type == "grid":
        gridsearch_ridge = GridSearchCV(pipeline_ridge,grid_param_ridge,cv=5,n_jobs=-1,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = gridsearch_ridge.fit(df_train_x,df_train_y)
    
    return best_model


# Elastic net regression pipeline
def elnet_pl(df_train_x,df_train_y,hyperparams,rand_s_cv=1):
    pipeline_elnet=Pipeline([('scaler',StandardScaler()),
                             ('elnet_algo',ElasticNet(random_state=rand_s_cv))])
    
    grid_param_elnet =  {"elnet_algo__alpha": hyperparams['reg_val'],
                         "elnet_algo__l1_ratio": hyperparams['l1_ratio']}
    
    if search_type == "random":
        randomsearch_elnet = RandomizedSearchCV(pipeline_elnet,grid_param_elnet,cv=5,n_jobs=-1,n_iter=10,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = randomsearch_elnet.fit(df_train_x,df_train_y.values.ravel())
    
    elif search_type == "grid":
        gridsearch_elnet = GridSearchCV(pipeline_elnet,grid_param_elnet,cv=5,n_jobs=-1,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = gridsearch_elnet.fit(df_train_x,df_train_y)
    
    return best_model


# Random forest regressor pipeline
def rfreg_pl(df_train_x,df_train_y,hyperparams,search_type="random",rand_s_cv=1):
    pipeline_rfreg=Pipeline([('scaler',StandardScaler()),
                             ('rfreg_algo',RandomForestRegressor(random_state=rand_s_cv))])
    
    grid_param_rfreg = {'rfreg_algo__n_estimators': hyperparams['n_estimators'],
                        'rfreg_algo__max_features': hyperparams['max_features'],
                        'rfreg_algo__max_depth': hyperparams['max_depth'],
                        'rfreg_algo__min_samples_split': hyperparams['min_samples_split'],
                        'rfreg_algo__min_samples_leaf': hyperparams['min_samples_leaf'],
                        'rfreg_algo__bootstrap': hyperparams['bootstrap']}
    
    if search_type == "random":
        randomsearch_rfreg = RandomizedSearchCV(pipeline_rfreg,grid_param_rfreg,cv=5,n_jobs=-1,n_iter=10,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = randomsearch_rfreg.fit(df_train_x,df_train_y.values.ravel())
    
    elif search_type == "grid":
        gridsearch_rfreg = GridSearchCV(pipeline_rfreg,grid_param_rfreg,cv=5,n_jobs=-1,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = gridsearch_rfreg.fit(df_train_x,df_train_y.values.ravel())
    
    return best_model


# Combined search pipeline
def comb_pl(df_train_x,df_train_y,hyperparams,search_type="random",rand_s_cv=1):
    pipeline_comb = Pipeline([('scaler',StandardScaler()),
                              ("algo", Lasso())])
    
    grid_param_comb = [
                        {'algo': [RandomForestRegressor(random_state=rand_s_cv)],
                        'algo__n_estimators': hyperparams['n_estimators'],
                        'algo__max_features': hyperparams['max_features'],
                        'algo__max_depth': hyperparams['max_depth'],
                        'algo__min_samples_split': hyperparams['min_samples_split'],
                        'algo__min_samples_leaf': hyperparams['min_samples_leaf'],
                        'algo__bootstrap': hyperparams['bootstrap']
                        },
                        {'algo': [ElasticNet(random_state=rand_s_cv)],
                        "algo__alpha": hyperparams['reg_val'],
                        "algo__l1_ratio": hyperparams['l1_ratio']
                        },
                        {'algo': [Lasso(random_state=rand_s_cv)],
                        "algo__alpha": hyperparams['reg_val']
                        },
                        {'algo': [Ridge(random_state=rand_s_cv)],
                          "algo__alpha": hyperparams['reg_val']
                          },
                        {'algo': [LinearRegression()]
                         }
                        ]
    
    if search_type == "random":
        randomsearch_comb = RandomizedSearchCV(pipeline_comb,grid_param_comb,cv=5,n_jobs=-1,n_iter=10,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = randomsearch_comb.fit(df_train_x,df_train_y.values.ravel())
    
    elif search_type == "grid":
        gridsearch_comb = GridSearchCV(pipeline_comb,grid_param_comb,cv=5,n_jobs=-1,scoring="neg_mean_squared_error",return_train_score=True)
        best_model = gridsearch_comb.fit(df_train_x,df_train_y.values.ravel())

    return best_model             

########################### INITIALIZING INPUTS ##############################

if __name__ == '__main__':
    ##default inputs
    pick_best_model = False
    search_type = "random"
    n_iter = 10
    regtype = "ridge"
    rand_s_cv = 1

    ##default hyperparameter inputs
    reg_val = [int(x) for x in np.logspace(0,3,num=10)]
    l1_ratio = [0.1,0.3,0.5,0.7,0.9]
    n_estimators = [int(x) for x in np.linspace(100,1000,num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10,100,num=10)]
    min_samples_split = [int(x+1) for x in np.logspace(0,2,num=4)]
    min_samples_leaf = [2,5,20,50]
    bootstrap = [True,False]

else:
    with open(argv[1],'r') as file:
        mlp_dict = (json.loads(file.read()))['mlp_dict']
        file.close()
    with open(argv[1],'r') as file:
        hyperparam_dict = (json.loads(file.read()))['hyperparameter_dict']
        file.close()
        
    ##import inputs from config file   
    pick_best_model = mlp_dict['pick_best_model']
    search_type = mlp_dict['search_type']
    n_iter = mlp_dict['n_iter']
    regtype = mlp_dict['regtype']
    rand_s_cv = mlp_dict['rand_s_cv']
    
    ##hyperparameter inputs
    reg_val = hyperparam_dict['reg_val']
    l1_ratio = hyperparam_dict['l1_ratio']    
    n_estimators = hyperparam_dict['n_estimators']
    max_features = hyperparam_dict['max_features']
    max_depth = hyperparam_dict['max_depth']
    min_samples_split = hyperparam_dict['min_samples_split']
    min_samples_leaf = hyperparam_dict['min_samples_leaf']
    bootstrap = hyperparam_dict['bootstrap']

######################## HYPERPARAMETER GRID SET-UP ##########################

hyperparams = {"reg_val": reg_val,
                "l1_ratio": l1_ratio,
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "bootstrap": bootstrap}

######################## MODEL TRAINING & EVALUATION #########################

kw_dict = {'random':'RandomizedSearchCV',
           'grid':'GridSearchCV',
           'ridge':'Ridge Regression',
           'lasso':'Lasso Regression',
           'linear':'Linear Regression',
           'elnet':'Elastic Net Regression',
           'rfreg':'Random Forest Regression'}

if search_type == "random":
    n_iter_ = str(n_iter)+" iterations of "
else:
    n_iter_ = ''

df_train_x = load("mlp/training_data_features")
df_test_x = load("mlp/test_data_features")
df_train_y = load("mlp/training_data_targets")
df_test_y = load("mlp/test_data_targets")

for var in df_train_y.columns:
    #as we have 2 target variables, we will use a 'for' loop to repeat the process
    df_train_y_ = df_train_y[[var]]
    df_test_y_ = df_test_y[[var]]
        
    if pick_best_model == True:
        print("______________________________________________________________________")
        print()
        print("Running "+n_iter_+kw_dict[search_type]+" across all algorithms for target variable '"+var+"'...")
        print()    
        
        model = comb_pl(df_train_x,df_train_y_,hyperparams)

        print("For target variable '"+var+"':")
        print()
        print("The best model & its set of hyperparameters are:")
        print(model.best_params_['algo'])
        print()
        print("Train Mean Squared Error:")
        print((-1)*(model.cv_results_['mean_train_score'][model.best_index_]))
        print()
        print("Cross-validation Mean Squared Error:")
        print((-1)*model.best_score_)

        y_pred = model.predict(df_test_x)
        
        test_mse = mean_squared_error(df_test_y_,y_pred)
        test_r2 = r2_score(df_test_y_,y_pred)
        test_adjr2 = 1-((1-test_r2)*((len(df_test_y_)-1)/(len(df_test_y_)-len(df_test_x.columns)-1)))
        
        print()          
        print("Test Mean Squared Error:")
        print(test_mse)        
        print()
        print("Test Adjusted R-Squared:")
        print(test_adjr2)
        
        model_filename = var+"_model"

    else:            
        print("______________________________________________________________________")   
        print()
        print("Running "+n_iter_+kw_dict[search_type]+" using "+kw_dict[regtype]+" algorithm for target variable '"+var+"'...")
        print()
        
        if regtype == "linear":
            model = linear_pl(df_train_x,df_train_y_)
        elif regtype == "lasso":
            model = lasso_pl(df_train_x,df_train_y_,hyperparams)
        elif regtype == "ridge":
            model = ridge_pl(df_train_x,df_train_y_,hyperparams)      
        elif regtype == "elnet":
            model = elnet_pl(df_train_x,df_train_y_,hyperparams)
        elif regtype == "rfreg":
            model = rfreg_pl(df_train_x,df_train_y_,hyperparams)
    
        print("With "+kw_dict[regtype]+", for target variable '"+var+"':")
        print()
        
        if regtype != "linear":  
            print("The best set of hyperparameters are:")
            print(model.best_params_)
            print()
            print("Train Mean Squared Error:")
            print((-1)*(model.cv_results_['mean_train_score'][model.best_index_]))
            print()
            print("Cross-validation Mean Squared Error:")
            print((-1)*model.best_score_)
        else:       
            y_pred = model.predict(df_train_x)
            
            train_mse = mean_squared_error(df_train_y_,y_pred)
    
            print("Train Mean Squared Error:")
            print(train_mse)
        
        y_pred = model.predict(df_test_x)
        
        test_mse = mean_squared_error(df_test_y_,y_pred)
        test_r2 = r2_score(df_test_y_,y_pred)
        test_adjr2 = 1-((1-test_r2)*((len(df_test_y_)-1)/(len(df_test_y_)-len(df_test_x.columns)-1)))
        
        print()          
        print("Test Mean Squared Error:")
        print(test_mse)       
        print()
        print("Test Adjusted R-Squared:")
        print(test_adjr2)
    
    model_filename = var+"_model"
    
    print()
    print("Saving model as '"+model_filename+"'...")
    with open("mlp/"+model_filename,'wb') as f:
        dump(model,f)
        f.close()
    print("'"+model_filename+"' saved.")
    print()

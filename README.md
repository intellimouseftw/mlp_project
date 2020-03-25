
## DESCRIPTION ##
This README serves as a guide and documentation on the usage of the machine learning pipeline created for the scooter rental dataset.

The MLP has been initialized for these following ML algorithms in mind:
1) Linear Regression
2) Lasso Regression
3) Ridge Regression
4) ElasticNet Regression
5) Random Forest Regression


## MAIN FILES ##

### run.sh
----------
Bash script which compiles the python modules and run them together. 
There are two modes of operation for the script:

1) Ingests data from a source file/link and passes it through a python module that fits the data to a desired ML model.
Configured for Randomized Search and Grid Search for hyperparameter tuning. Can be within a specific ML algorithm or across all configured algorithms.

2) Predicts the target variables given new instances. New instances has to be passed in through a csv file, in the same format as the original dataset.

Running the script will always prompt the user for an input on which mode of operation is desired.


### config.json
---------------
JSON file that holds dictionaries containing available options to run the MLP in a desired manner.
Description of the dictionaries are as follows:
1) "data_source": Specifies the source for the dataset (can be in either url or filepath)

2) "preprocessing_dict": Dictionary holding the arguments to be passed down within the data pre-processing step.
	* (str variable) cat_label  - labels of categorical variables (excluding binary variables) to be used in the model, separated by commas without spacing.
	* (str variable) cont_bin_label  - labels of continuous and binary variables to be used in the model, separated by commas without spacing.
	* (str variable) tar_label - labels of target variables to be used in the model, separated by commas without spacing.
	* (float variable) test_sz - specifies the ratio of data split from the main dataset to be used as test data for model metrics purposes.
	* (int variable) rand_s_split - specifies the random state used for the train-test-split method in scikit-learn.

3) "hyperparameter_dict": Dictionary holding lists that specifies the hyperparameter tuning range.
	* reg_val - Regularization parameter for Lasso, Ridge, ElasticNet algorithms
	* l1_ratio - Scale of l1 penalty for ElasticNet regressions
	* n_estimators - Number of trees in random forest
	* max_features - Number of features to consider at every split
	* max_depth - Maximum number of levels in tree
	* min_samples_split - Minimum number of samples required to split a node
	* min_samples_leaf - Minimum number of samples required at each leaf node
	* bootstrap - Method of selecting samples for training each tree (with replacement/without replacement)

4) "mlp_dict": Dictionary holding arguments to be passed down within the mlp step.
	* (bool variable) pick_best_model - if True, hyperparameter tuning will perform search across all algorithms. if False, user has to specify "regtype". MLP will only perform hyperparameter tuning over the specified reg_type algorithm.
	* (str variable) search_type - select between "random" and "grid". Specifies the type of search to perform, RandomizedSearchCV vs GridSearchCV. Do note that if "grid" is chosen for (pick_best_model = True) or (regtype = rfreg), it will be a very computationally expensive process.
	* (int variable) n_iter - specifies the number of random search iterations to carry out.
	* (str variable) reg_type - if (pick_best_model = False), then the mlp will perform hyperparameter tuning over the specified reg_type algorithm.

| regtype       | Algorithm         |
| ------------- |-------------------|
| 'linear'      | Linear Regression |
| 'lasso'       | Lasso Regression  |
| 'ridge'       | Ridge Regression  | 
| 'elnet'       | Elastic Net Reg   |
| 'rfreg'       | Random Forest Reg |
|-----------------------------------|
	* (int variable) rand_s_cv - specifies the random state used for stratified k-fold splits when performing hyperparameter tuning.

5) "instance_file": Specifies the file name for the new instances to perform predictions on (MUST be a csv file placed in "predictions" folder)

6) "prediction_dist": Dictionary holding arguments to be passed down within the predictions step.
	* (bool variable) save_pred - specifies if user wishes to save the predictions into a csv file.
	* (str variable) pred_filename - if save_pred = True, this specifies the filename that the predictions will be saved onto.


## USAGE ##
Tweak config.json to the desired settings, then execute run.sh script.

A default config file with the best algorithm and tuning parameters, along with the best models (saved as scikit-learn objects) can be found in the "best model" folder.

To use the best model, copy and replace "cbrt_guest-users_model" & "cbrt_registered-users_model" from "best model" folder into "mlp" folder.
The best model can then be used to perform predictions on new instances using operation mode 2 for run.sh.

To obtain the metrics for the best model, copy and replace "config.json" from "best model" folder to the root folder, and execute run.sh on operation mode 1.


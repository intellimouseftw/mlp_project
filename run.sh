#!/bin/bash

echo ------------------
echo Select mode [1]/[2]:
echo ------------------
echo [1]: Perform training on dataset and obtain desired ML model
echo [2]: Generate predictions for new instances
echo
echo NOTE:
echo - Ensure desired model fitting and training is performed with mode [1] before using mode [2]. More details can be found in the README.
echo - The MLP takes in inputs from "config.json". Ensure config file is set-up properly before using script. Documentation can be found in the README. 
echo
echo INPUT:
read choice

if [ "$choice" == 1 ]; then
	python3 mlp/main.py config.json train
else
	if [ "$choice" == 2 ]; then
		python3 mlp/main.py config.json predict
	else
		echo Only inputs [1] or [2] are accepted. Ending script...
		echo
	fi
fi

from sklearn.svm import LinearSVR
from typing import Union, List, Dict, Tuple
from collections import defaultdict
import time 


import sys 
import os 
import argparse
import yaml 
import tqdm

import numpy as np
import pickle
import pandas as pd

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import  StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


import io_utils, dataset_utils, model_utils

import GraphTransforms as gt

from tdc.single_pred import ADME




def main(
	config:Dict
	) -> None:
	
	
	print("\n")
	short_name, tdc_name, tdc_path, short_size, long_size =  \
		io_utils.unpack_parameters(config['DATASET_PARAMS'])

	num_trials, test_pct, rngseed, model_name, exp_type, score, output_path = \
	 	io_utils.unpack_parameters(config['EXPERIMENT_PARAMS'])
	
	alpha_logspace, alpha_min, alpha_max, num_alphas, num_l1s, l1_min, l1_max =\
	 	io_utils.unpack_parameters(config['CV_PARAMS'])

	feature_type, standardize, numScales_v, maxMoment_v, central_v, numScales_e, maxMoment_e, central_e  = \
		io_utils.unpack_parameters(config['FEATURE_PARAMS'])

	opath = "{b}/{t}/{d}/{m}/".format(b=output_path,t=exp_type,d=short_name,m=model_name.upper())
	os.makedirs(opath,exist_ok = True)

	model, param_grid = model_utils.make_model_and_param_grid(model_name, standardize, "regr", config['MODEL_PARAMS'])


	data = ADME(name = tdc_name, path = tdc_path) # will download if not already present
	dataframe = data.get_data()

	rng = np.random.RandomState(seed = rngseed)
	


	### process dataset to graph representations
	
	msg = "Making Dataset"
	io_utils.star_echo("Making Dataset Dictionary")
	
	start = time.time()
	
	converted_DataSet = dataset_utils.make_dataset(dataframe,short_size,long_size)
	obs_index = list(converted_DataSet.keys())

	end = time.time()

	io_utils.star_echo("took {t} seconds to make it".format(t = str(end-start)))

	results = defaultdict(list)

	if feature_type.lower() == "fingerprints":
		feature_types = dataset_utils.FINGERPRINT_FEATURES
	else:
		feature_types = [feature_type]

	for feature_type in feature_types:
		start = time.time()

		for i in tqdm.tqdm(range(num_trials)):

			trn_idx, test_idx = train_test_split(obs_index ,random_state = rng)
			train_ds = {i:converted_DataSet[i] for i in trn_idx}
			test_ds = {i:converted_DataSet[i] for i in test_idx}
			X_train, y_train = dataset_utils.make_numpy_dataset(train_ds,feature_type, numScales_v, maxMoment_v, central_v)
			X_test, y_test  = dataset_utils.make_numpy_dataset(test_ds,feature_type, numScales_v, maxMoment_v, central_v)


			regressor = GridSearchCV(model, param_grid)
			regressor.fit(X_train, y_train)
			preds = regressor.best_estimator_.predict(X_test)
			
			MAE  = mean_absolute_error(y_test,preds)
			MSE = mean_squared_error(y_test, preds)
			RMSE = mean_squared_error(y_test,preds, squared = False)

			results['iter'].append(i)
			results['feature'].append(feature_type)
			results['MAE'].append(MAE)
			results['MSE'].append(MSE)
			results['RMSE'].append(RMSE)
		

		end = time.time()
		io_utils.star_echo("Took {t} seconds per iteration".format(t=(end-start)/num_trials))

		df = pd.DataFrame(results)
		df.to_csv("{p}/{f}".format(p=opath,f=feature_type))






		
		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.config) as file:
		config = yaml.safe_load(file)
	
	main(config)
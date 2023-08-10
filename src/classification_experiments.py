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

	model, param_grid = model_utils.make_model_and_param_grid(model_name, standardize, "regr",config['MODEL_PARAMS'])


	data = ADME(name = tdc_name, path = tdc_path) # will download if not already present
	dataframe = data.get_data()

	rng = np.random.RandomState(seed = rngseed)


	




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.config) as file:
		config = yaml.safe_load(file)
	
	main(config)
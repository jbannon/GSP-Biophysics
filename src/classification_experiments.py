from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVR
from typing import Union, List, Dict, Tuple
from collections import defaultdict
import time 
from sklearn.metrics import accuracy_score

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
	
	
	num_trials, test_pct, rngseed, short_size, long_size, model_list, prefix, exp_type, score, output_path,tdc_path = \
	 	io_utils.unpack_parameters(config['EXPERIMENT_PARAMS'])

	feature_type, numScales_v, maxMoment_v, central_v, numScales_e, maxMoment_e, central_e  = \
		io_utils.unpack_parameters(config['FEATURE_PARAMS'])


	datasets  = dataset_utils.dataset_dict[exp_type.lower()]

	rng = np.random.RandomState(seed = rngseed)

	

	if feature_type.lower() == "fingerprints":
		feature_types = dataset_utils.FINGERPRINT_FEATURES
	elif feature_type.lower() == 'all':
		feature_types = dataset_utils.FINGERPRINT_FEATURES + ['DW', 'DWLG']

	else:
		feature_types = [feature_type]



	for dataset in datasets:

		io_utils.star_echo("Working on {ds}".format(ds=dataset))
		short_name = dataset_utils.dataset_to_short_name[dataset]
		if dataset == 'PAMPA_APPROVED':
			data = ADME(name = 'PAMPA_NCATS',path = tdc_path)
			dataframe = data.get_approved_set()

		else:
			data = ADME(name = dataset, path = tdc_path) # will download if not already present
			dataframe = data.get_data()


		io_utils.star_echo("Making Dataset Dictionary")
			
		start = time.time()
		
		converted_DataSet = dataset_utils.make_dataset(dataframe,short_size,long_size)
		obs_index = list(converted_DataSet.keys())

		end = time.time()

		io_utils.star_echo("took {t} seconds to make it".format(t = str(end-start)))

		for model_name in model_list:
			
			io_utils.star_echo("Working on {m}".format(m = model_name))
			opath = "{b}/{t}/{d}/{m}/".format(b=output_path,t=exp_type,d=short_name,m=model_name.upper())
			os.makedirs(opath,exist_ok = True)
			
			for feature_type in feature_types:
				
				start = time.time()

				# standardize_features = False if feature_type.lower() in dataset_utils.FINGERPRINT_FEATURES else False
				standardize_features = True
				model, param_grid = model_utils.make_model_and_param_grid(model_name, 
						standardize_features,
						 prefix, 
						 config['MODEL_PARAMS'][model_name.upper()],
						 rng)
				
				results = defaultdict(list)

				io_utils.star_echo("working on feature: {f}".format(f=feature_type))

				for i in tqdm.tqdm(range(num_trials)):

					trn_idx, test_idx = train_test_split(obs_index ,random_state = rng)
					train_ds = {i:converted_DataSet[i] for i in trn_idx}
					test_ds = {i:converted_DataSet[i] for i in test_idx}
					X_train, y_train = dataset_utils.make_numpy_dataset(train_ds,feature_type, numScales_v, maxMoment_v, central_v,
						numScales_e, maxMoment_e, central_e)
					X_test, y_test  = dataset_utils.make_numpy_dataset(test_ds,feature_type, numScales_v, maxMoment_v, central_v,
						numScales_e, maxMoment_e, central_e)


					classifier = GridSearchCV(model, param_grid)
					classifier.fit(X_train, y_train)
					preds = classifier.best_estimator_.predict(X_test)
					
					
					acc = accuracy_score(y_test, preds)
					tn, fp, fn, tp = confusion_matrix(y_test, preds,labels = [0,1]).ravel()
					results['iter'].append(i)
					results['feature'].append(feature_type)

					results['acc'].append(acc)
					results['tp'].append(tp)
					results['tn'].append(tn)

					results['fp'].append(fp)
					results['fn'].append(fn)

				

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
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import sys 
import os 
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml 
from typing import Union, List, Dict, Tuple
import numpy as np
import pickle
from sklearn.model_selection import LeaveOneOut,GridSearchCV, KFold, StratifiedKFold, train_test_split
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tqdm
import GraphTransforms

FINGERPRINT_FEATURES = ['morgan_short','morgan_long','top_short','top_long','bit_short','bit_long']

def make_numpy_dataset(
	data:Dict,
	feature_type:str
	):
	X, y = [np.array([]) for i in range(2)]
	if feature_type in FINGERPRINT_FEATURES:
		for k in data.keys():
			molecule_data = data[k]
			X = np.vstack( (X, molecule_data[feature_type])) if X.size else molecule_data[feature_type]
			y = np.vstack( (y, molecule_data['y'])) if  y.size else np.array(molecule_data['y'])
	return X, y


def main(
	config:Dict
	) -> None:
	
	with open(config['ds_base']+config['dataset']+"/processed.pickle",'rb') as f:
		DS = pickle.load(f)
	
	obs_index = list(DS.keys())
	rng = np.random.RandomState(seed = config['rngseed'])
	feature_type = config['feature_type']
	results = defaultdict(list)

	if feature_type.lower() == "fingerprints":
		feature_types = FINGERPRINT_FEATURES
	else:
		feature_types = [feature_type]
	
	params = {'alpha':np.logspace(10**-2, 10**2, 10), 'l1_ratio':np.linspace(0.01, 0.99,10)}
	regr = ElasticNet(random_state = rng)
	
	for feature_type in feature_types:
		for i in tqdm.tqdm(range(config['num_trials'])):
			trn_idx, test_idx = train_test_split(obs_index ,random_state = rng)
			train_ds = {i:DS[i] for i in trn_idx}
			test_ds = {i:DS[i] for i in test_idx}
			X_train, y_train = make_numpy_dataset(train_ds,feature_type)
			X_test, y_test  = make_numpy_dataset(test_ds,feature_type)

			
			clf = GridSearchCV(regr, params)
			clf.fit(X_train, y_train)
			preds = clf.best_estimator_.predict(X_test)
			
			MAE  = mean_absolute_error(y_test,preds)
			MSE = mean_squared_error(y_test, preds)
			RMSE = mean_squared_error(y_test,preds, squared = False)

			results['iter'].append(i)
			results['feature'].append(feature_type)
			results['MAE'].append(MAE)
			results['MSE'].append(MSE)
			results['RMSE'].append(RMSE)



		df = pd.DataFrame(results)
		opath = "{b}/{t}/{d}".format(b=config['output_path'],t=config['exp_type'],d=config['dataset'])
		os.makedirs(opath,exist_ok = True)
		df.to_csv("{p}/{f}".format(p=opath,f=feature_type))


		
		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.config) as file:
		config = yaml.safe_load(file)
	
	main(config)
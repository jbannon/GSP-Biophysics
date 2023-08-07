from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, QuantileRegressor
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
	
	thresh = 0.2
	
	upper_regressor = QuantileRegressor(quantile = 1 - thresh/2,solver='highs')
	lower_regressor = QuantileRegressor(quantile = thresh/2,solver = 'highs')
	params = {'alpha':np.logspace(10**-2, 10**2, 10)}
	
	cal_size = 500

	for feature_type in feature_types:
		for i in tqdm.tqdm(range(config['num_trials'])):
			trn_idx, test_idx = train_test_split(obs_index ,random_state = rng)
			
			trn_idx, cal_idx = train_test_split(trn_idx,test_size = cal_size)

			train_ds = {i:DS[i] for i in trn_idx}
			cal_ds = {i:DS[i] for i in cal_idx}
			test_ds = {i:DS[i] for i in test_idx}
			
			X_train, y_train = make_numpy_dataset(train_ds,feature_type)
			X_cal, y_cal = make_numpy_dataset(cal_ds,feature_type)

			X_test, y_test  = make_numpy_dataset(test_ds,feature_type)

			
			regr_U = GridSearchCV(upper_regressor, params)
			regr_U.fit(X_train,y_train.squeeze(axis=1))

			regr_L = GridSearchCV(lower_regressor, params)
			regr_L.fit(X_train,y_train.squeeze(axis=1))


			upperQs = regr_U.best_estimator_.predict(X_cal).reshape(-1,1)
			lowerQs = regr_L.best_estimator_.predict(X_cal).reshape(-1,1)

			
			
			L = lowerQs - y_cal
			U = y_cal - upperQs
			
			scores = np.amax(np.hstack((L,U)),axis=1)
			

			q = np.ceil((1-thresh)*(cal_size+1))/cal_size


			q_hat = np.quantile(scores,q)
			
			U_test = regr_U.best_estimator_.predict(X_test) + q_hat
			L_test = regr_L.best_estimator_.predict(X_test) - q_hat
			
			print(U_test)
			print(L_test)
			print(y_test.shape)

			c = 0
			for i in range(y_test.shape[0]):
				if y_test[i,0] >= L_test[i] and y_test[i,0]<= U_test[i]:
					c+=1
			print(c)
			print(c/y_test.shape[0])
			sys.exit(1)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.config) as file:
		config = yaml.safe_load(file)
	
	main(config)
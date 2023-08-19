# from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
import numpy as np 
from typing import Union, List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def make_model(
	model_name:str,
	standardize: bool,
	prefix:str,
	rng = None
	) ->List[Tuple]:
	

	if model_name == 'OLS':
		model = (prefix,LinearRegression())
	elif model_name == 'Ridge':
		model = (prefix,Ridge())
	elif model_name == 'SVR':
		model = (prefix, LinearSVR(max_iter = 100000))
	elif model_name == 'RFR':
		model = (prefix, RandomForestRegressor(random_state = rng,min_samples_leaf = 0.05))
	elif model_name == 'SVC':
		model = (prefix, LinearSVC(dual = 'auto',max_iter = 100000, random_state = rng))
	elif model_name == 'KNN':
		model = (prefix, KNeighborsClassifier())
	elif model_name == 'RBF':
		model = (prefix, SVC(kernel = 'rbf'))
	elif model_name == 'LR':
		model = (prefix, LogisticRegression())

	if standardize:
		model = Pipeline([('scaler', StandardScaler()),model])
	else:
		model = Pipeline([model])
	return model



def make_integer_param_range(params:Dict):

	param_vals = np.arange(params['min'],params['max'],params['step'])
	return param_vals


def make_float_param_range(params:Dict):
	if params['logspace']:
		param_vals = np.logspace(params['min'],params['max'],params['step'])
	else: 
		param_vals = np.linspace(params['min'],params['max'],params['step'])

	return param_vals


def make_model_and_param_grid(
	model_name:str,
	standardize:bool,
	prefix: str,
	param_dict: Dict,
	rng = None
	):
	

	model = make_model(model_name,standardize,prefix,rng = None)

	param_grid = {}
	if model_name.upper() != "OLS":

		for param_name in param_dict.keys():

			param_vals = param_dict[param_name]
			
			if param_vals['type'] == "none":
				continue
			if param_vals['type'] == "int":
				vals = make_integer_param_range(param_vals)
			elif param_vals['type'] == "float":
				
				vals = make_float_param_range(param_vals)
			elif param_vals['type'] == "list":
				vals = param_vals['values']


		
			param_grid[prefix+"__" + param_name] = vals
	else:
		param_grid = {}

	return model, param_grid



	





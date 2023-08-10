# from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
import numpy as np 
from typing import Union, List, Dict, Tuple




def make_model(
	model_name:str,
	standardize: bool,
	prefix:str,
	) ->List[Tuple]:
	

	if model_name == 'OLS':
		model = (prefix,LinearRegression())
	elif model_name == 'Ridge':
		model = (prefix,Ridge())
	elif model_name == 'SVR':
		model = (prefix, LinearSVR(max_iter = 100000))

	if standardize:
		model = Pipeline([('scaler', StandardScaler()),model])
	else:
		model = Pipeline([model])
	return model


def make_model_and_param_grid(
	model_name:str,
	standardize:bool,
	prefix: str,
	param_dict: Dict
	):
	

	model = make_model(model_name,standardize,prefix)

	param_grid = {}
	for name, minval,maxval, npoints, logspace in  zip(param_dict['names'], 
		param_dict['mins'],
		 param_dict['maxs'],
		 param_dict['steps'],
		 param_dict['logspaces']
		 ):

		if logspace:
			vals = np.logspace(minval, maxval, npoints)
		else:
			vals = np.linspace(minval, maxval, npoints)
		param_grid[prefix+"__"+name] = vals

	return model, param_grid



	





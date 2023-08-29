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

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import  StandardScaler


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, KFold

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, confusion_matrix, accuracy_score,f1_score

from sklearn.decomposition import PCA

import io_utils, dataset_utils, model_utils

import GraphTransforms as gt

from tdc.single_pred import ADME

from sklearn.neighbors import KNeighborsClassifier


def main():
	dataset = 'Lipophilicity_AstraZeneca'
	

	MAX_VERTEX_SCALES = 6
	MAX_VERTEX_MOMENTS = 6
	MAX_EDGE_SCALES = 6
	MAX_EDGE_MOMENTS = 6

	seed = 1234

	tdc_path = "../data/raw/tdc"
	short_size = 32
	long_size = 1024

	prefix = 'regr'
	pca_components = 10

	model_tuple = (prefix,Ridge())
	grid = {prefix + "__alpha":np.logspace(-2,2,50)}



	model = Pipeline([('scaler',StandardScaler()),model_tuple])

	pca_model = Pipeline(
		[('scaler',StandardScaler()),
		 ('dimred',PCA(n_components = pca_components)),
		model_tuple
		])


	data = ADME(name = dataset, path = tdc_path) # will download if not already present
	dataframe = data.get_data()	
	data_set = dataset_utils.make_dataset(dataframe,short_size,long_size)

	obs_index = list(data_set.keys())
	path = "../results/sensitivity/regression/"
	os.makedirs(path,exist_ok=True)

	short_name = dataset_utils.dataset_to_short_name[dataset]

	
	rng = np.random.RandomState(1234)
	results = defaultdict(list)	
	feature_type = 'DWLG'
	n_folds = 20

	splitter = KFold(shuffle = True, random_state = rng)

	for max_vertex_scale in range(1,MAX_VERTEX_SCALES):
			for max_vertex_moment in range(1,MAX_VERTEX_MOMENTS):
				for center_vertex_features in [True, False]:
					for max_edge_scale in range(1, MAX_EDGE_SCALES):
						for max_edge_moment in range(1,MAX_EDGE_MOMENTS):
							for center_edge_features in [True, False]:




								io_utils.star_echo("\nWorking on:\n\tvertex scale {J}\n\tvertex moment {p}\n\tcentering: {c}\n".format(J=max_vertex_scale,p = max_vertex_moment, c= center_vertex_features))
								
								X,y = dataset_utils.make_numpy_dataset( {i:data_set[i] for i in obs_index},feature_type, 
								max_vertex_scale, max_vertex_moment, center_vertex_features,
								numScales_e = max_edge_scale, maxMoment_e = max_edge_moment, central_e = center_edge_features
								)

								for i, (train_idx, test_idx) in tqdm.tqdm(enumerate(splitter.split(X,y))):
									
									X_train, y_train = X[train_idx,:], y[train_idx]
									X_test, y_test = X[test_idx,:], y[test_idx]


									
									cv_model = GridSearchCV(model, grid)
									cv_model.fit(X_train, y_train)
									

									

									results['iter'].append(i)
									results['max_vertex_scale'].append(max_vertex_scale)
									results['max_vertex_moment'].append(max_vertex_moment)
									results['centered_vertex_features'].append(center_vertex_features)
									results['max_edge_scale'].append(max_vertex_scale)
									results['max_edge_moment'].append(max_vertex_moment)
									results['centered_edge_features'].append(center_vertex_features)
									
									results['pca'].append("No")

									
									
									preds = cv_model.best_estimator_.predict(X_test)
									MAE  = mean_absolute_error(y_test,preds)
									MSE = mean_squared_error(y_test, preds)
									RMSE = mean_squared_error(y_test,preds, squared = False)
								
									results['MAE'].append(MAE)
									results['MSE'].append(MSE)
									results['RMSE'].append(RMSE)



	results = pd.DataFrame(results)

	results.to_csv(path + "{ft}.csv".format(ft=feature_type), index = False)
		
if __name__ == '__main__':
	main()


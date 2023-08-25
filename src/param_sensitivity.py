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
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, confusion_matrix, accuracy_score,f1_score

from sklearn.decomposition import PCA

import io_utils, dataset_utils, model_utils

import GraphTransforms as gt

from tdc.single_pred import ADME

from sklearn.neighbors import KNeighborsClassifier


def make_model_and_grid(
	task:str,
	prefix:str,
	pca_components:int = 10
	):

	if task == 'regression':
		model_tuple = (prefix,Ridge())
		grid = {prefix + "__alpha":np.logspace(-2,2,500)}
	elif task == 'classification':
		model_tuple = (prefix, KNeighborsClassifier())
		grid = {prefix + "__n_neighbors":np.arange(2,20)}


	model = Pipeline([('scaler',StandardScaler()),model_tuple])
	
	pca_model = Pipeline(
		[('scaler',StandardScaler()),
		 ('dimred',PCA(n_components = pca_components)),
		model_tuple
		])

	return model, pca_model, grid



					




def run_lg_experiments(task: str,
	feature_type:str,
	num_trials:int,
	data_set:Dict, 
	obs_index:List[int],
	MAX_VERTEX_SCALES:int,
	MAX_VERTEX_MOMENTS:int,
	MAX_EDGE_SCALES: int,
	MAX_EDGE_MOMENTS:int,
	prefix: str):
	

	if task == 'regression':
		model, pca_model, grid  = make_model_and_grid(task, prefix)
	elif task == 'classification':
		model, pca_model, grid  = make_model_and_grid(task, prefix)


	rng = np.random.RandomState(1234)
	results = defaultdict(list)
	for max_vertex_scale in range(1,MAX_VERTEX_SCALES):
			for max_vertex_moment in range(1,MAX_VERTEX_MOMENTS):
				for center_vertex_features in [True, False]:
					for max_edge_scale in range(1,MAX_EDGE_SCALES):
						for max_edge_moment in range(1,MAX_EDGE_MOMENTS):
							for center_edge_features in [True,False]:
								for i in tqdm.tqdm(range(num_trials)):

									trn_idx, test_idx = train_test_split(obs_index ,random_state = rng)
									train_ds = {i:data_set[i] for i in trn_idx}
									test_ds = {i:data_set[i] for i in test_idx}

								X_train, y_train = dataset_utils.make_numpy_dataset(train_ds,feature_type, 
								max_vertex_scale, max_vertex_moment, center_vertex_features,
								numScales_e = max_edge_scale, maxMoment_e = max_edge_moment, central_e = center_edge_features
								)
								X_test, y_test  = dataset_utils.make_numpy_dataset(test_ds,feature_type,
								max_vertex_scale, max_vertex_moment, center_vertex_features,
								numScales_e = max_edge_scale, maxMoment_e = max_edge_moment, central_e = center_edge_features
								)
						
						cv_model = GridSearchCV(model, grid)
						cv_model.fit(X_train, y_train)
						

						

						results['iter'].append(i)
						results['max_vertex_scale'].append(max_vertex_scale)
						results['max_vertex_moment'].append(max_vertex_moment)
						results['centered_vertex_features'].append(center_vertex_features)

						results['max_edge_scale'].append(max_edge_scale)
						results['max_edge_moment'].append(max_edge_moment)
						results['centered_edge_features'].append(center_edge_features)
						results['pca'].append("No")

						if task == 'classification':
							preds = cv_model.best_estimator_.predict(X_test)
							pred_probs = cv_model.best_estimator_.predict_proba(X_test)
							acc = accuracy_score(y_test, preds)
							roc = roc_auc_score(y_test, pred_probs[:,1])
							tn, fp, fn, tp  = confusion_matrix(y_test, preds, labels = [0,1]).ravel()
							f1 = f1_score(y_test,preds)

							results['test_acc'].append(acc)
							results['test_roc'].append(roc)
							results['true negatives'].append(tn)
							results['false_negatives'].append(fn)
							results['true_positives'].append(tp)
							results['false_positives'].append(fp)
							results['f1'].append(f1)
						elif task == 'regression':
							preds = cv_model.best_estimator_.predict(X_test)
							MAE  = mean_absolute_error(y_test,preds)
							MSE = mean_squared_error(y_test, preds)
							RMSE = mean_squared_error(y_test,preds, squared = False)
						
							results['MAE'].append(MAE)
							results['MSE'].append(MSE)
							results['RMSE'].append(RMSE)


						if X_train.shape[1]>10:
							results['iter'].append(i)
							results['max_vertex_scale'].append(max_vertex_scale)
							results['max_vertex_moment'].append(max_vertex_moment)
							results['centered_vertex_features'].append(center_vertex_features)
							results['pca'].append("Yes")

							pca_cv = GridSearchCV(pca_model, grid)
							pca_cv.fit(X_train,y_train)

							if task == 'classification':
								preds = pca_cv.best_estimator_.predict(X_test)
								pred_probs = pca_cv.best_estimator_.predict_proba(X_test)
								
								acc = accuracy_score(y_test, preds)
								roc = roc_auc_score(y_test, pred_probs[:,1])
								tn, fp, fn, tp  = confusion_matrix(y_test, preds, labels = [0,1]).ravel()
								f1 = f1_score(y_test,preds)

								results['test_acc'].append(acc)
								results['test_roc'].append(roc)
								results['true negatives'].append(tn)
								results['false_negatives'].append(fn)
								results['true_positives'].append(tp)
								results['false_positives'].append(fp)
								results['f1'].append(f1)

							elif task == 'regression':
								preds = cv_model.best_estimator_.predict(X_test)
								MAE  = mean_absolute_error(y_test,preds)
								MSE = mean_squared_error(y_test, preds)
								RMSE = mean_squared_error(y_test,preds, squared = False)
						
								results['MAE'].append(MAE)
								results['MSE'].append(MSE)
								results['RMSE'].append(RMSE)


	results = pd.DataFrame(results)
	return results

	
def main():


	classification_name = 'PAMPA_NCATS'
	regression_name = 'Lipophilicity_AstraZeneca'
	num_trials = 20

	MAX_VERTEX_SCALES = 5
	MAX_VERTEX_MOMENTS = 5

	MAX_EDGE_SCALES  = 5
	MAX_EDGE_MOMENTS = 5
	tdc_path = "../data/raw/tdc"
	short_size = 32
	long_size = 1024





	for dataset, task in zip([classification_name, regression_name],['classification','regression']):
		

		short_name = dataset_utils.dataset_to_short_name[dataset]
		io_utils.star_echo("Working on {ds}".format(ds=short_name))

		data = ADME(name = dataset, path = tdc_path) # will download if not already present
		dataframe = data.get_data()	
		converted_DataSet = dataset_utils.make_dataset(dataframe,short_size,long_size)

		obs_index = list(converted_DataSet.keys())
		path = "../results/sensitivity/{t}/".format(t=task)
		os.makedirs(path,exist_ok=True)

		for feature_type in ['DWLG']:
			if feature_type == 'DW':
				results = run_vertex_experiments(task, feature_type,num_trials, converted_DataSet,obs_index, MAX_VERTEX_SCALES, MAX_VERTEX_MOMENTS,task[:4])
				
			elif feature_type == 'DWLG':
				results = run_lg_experiments(task, feature_type,num_trials, converted_DataSet,obs_index, 
					MAX_VERTEX_SCALES, MAX_VERTEX_MOMENTS,
					MAX_EDGE_SCALES, MAX_EDGE_MOMENTS,task[:4])
				results.to_csv(path+feature_type+".csv")



					

		


if __name__ == '__main__':
	main()
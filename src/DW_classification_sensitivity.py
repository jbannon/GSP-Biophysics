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

from sklearn.linear_model import  LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import  StandardScaler


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, confusion_matrix, accuracy_score,f1_score
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, auc 

from sklearn.decomposition import PCA

import io_utils, dataset_utils, model_utils

import GraphTransforms as gt

from tdc.single_pred import ADME

from sklearn.neighbors import KNeighborsClassifier



def main():
	dataset = 'PAMPA_NCATS'

	num_trials = 50

	MAX_VERTEX_SCALES = 5
	MAX_VERTEX_MOMENTS = 5

	tdc_path = "../data/raw/tdc"
	short_size = 32
	long_size = 1024

	prefix = 'clf'
	pca_components = 10

	# model_tuple = (prefix,Ridge())

	model_tuples = {'SVC':(prefix, LinearSVC()), 'RBF': (prefix, SVC(kernel = 'rbf',gamma = 'scale')), 'LR':(prefix, LogisticRegression())}


	svc_grid = {prefix + "__C":[10**i for i in range(-6,1)]}
	lr_grid = {prefix + "__C":[10**i for i in range(-6,-3)]}
	
	# sys.exit(1)


	scale_pre = [('scaler',StandardScaler())]
	pca_pre  = [('scaler',StandardScaler()),
		 ('dimred',PCA(n_components = pca_components))]


	simple_tuples = {k:Pipeline(scale_pre + [model_tuples[k]]) for k in model_tuples.keys()}
	pca_tuples = {k:Pipeline(pca_pre + [model_tuples[k]]) for k in model_tuples.keys()}
	

	# models = {'SVC':}




	data = ADME(name = dataset, path = tdc_path) # will download if not already present
	dataframe = data.get_data()	
	data_set = dataset_utils.make_dataset(dataframe,short_size,long_size)

	obs_index = list(data_set.keys())
	path = "../results/sensitivity/classification/"
	os.makedirs(path,exist_ok=True)

	short_name = dataset_utils.dataset_to_short_name[dataset]


	model_names = ['SVC','RBF','LR']
	
	rng = np.random.RandomState(1234)
	results = defaultdict(list)	
	feature_type = 'DW'
	for max_vertex_scale in range(4,MAX_VERTEX_SCALES):
			for max_vertex_moment in range(4,MAX_VERTEX_MOMENTS):
				for center_vertex_features in [True, False]:
					io_utils.star_echo("\nWorking on:\n\tvertex scale {J}\n\tvertex moment {p}\n\tcentering: {c}\n".format(J=max_vertex_scale,p = max_vertex_moment, c= center_vertex_features))
					for i in tqdm.tqdm(range(num_trials)):

						trn_idx, test_idx = train_test_split(obs_index ,random_state = rng)
						train_ds = {i:data_set[i] for i in trn_idx}
						test_ds = {i:data_set[i] for i in test_idx}

						X_train, y_train = dataset_utils.make_numpy_dataset(train_ds,feature_type, 
							max_vertex_scale, max_vertex_moment, center_vertex_features,
							numScales_e = None, maxMoment_e = None, central_e = None
							)
						X_test, y_test  = dataset_utils.make_numpy_dataset(test_ds,feature_type,
							max_vertex_scale, max_vertex_moment, center_vertex_features,
							numScales_e = None, maxMoment_e = None, central_e = None
							)
						
						for model in tqdm.tqdm(model_names,leave=False,desc = 'simple models'):
							if model in ['SVC','RBF']:
								grid = svc_grid
							else:
								grid = lr_grid
							
							cv_model = GridSearchCV(simple_tuples[model], grid)
							cv_model.fit(X_train, y_train)
						

						

							results['iter'].append(i)
							results['model'].append(model)
							results['max_vertex_scale'].append(max_vertex_scale)
							results['max_vertex_moment'].append(max_vertex_moment)
							results['centered_vertex_features'].append(center_vertex_features)
							results['pca'].append("No")

							
							
							preds = cv_model.best_estimator_.predict(X_test)
							acc = accuracy_score(y_test, preds)
							b_acc = balanced_accuracy_score(y_test,preds)

							results['acc'].append(acc)
							results['bal_acc'].append(b_acc)

							if model == 'LR':
								prob_preds = cv_model.best_estimator_.predict_proba(X_test)
								roc_auc = roc_auc_score(y_test, prob_preds[:,1])
								prec, rec, thresh = precision_recall_curve(y_test,prob_preds[:,1])
								pr_auc = auc(rec,prec)
								results['roc_auc'].append(roc_auc)
								results['pr_auc'].append(pr_auc)
							else:
								results['roc_auc'].append(-1)
								results['pr_auc'].append(-1 )


							

						if max_vertex_scale*max_vertex_moment>10:
							for model in tqdm.tqdm(model_names,leave = False,desc = 'pca models'):
								if model in ['SVC','RBF']:
									grid = svc_grid
								else:
									grid = lr_grid
							
								results['iter'].append(i)
								results['model'].append(model)
								results['max_vertex_scale'].append(max_vertex_scale)
								results['max_vertex_moment'].append(max_vertex_moment)
								results['centered_vertex_features'].append(center_vertex_features)
								results['pca'].append("Yes")

								cv_model = GridSearchCV(pca_tuples[model], grid)
								cv_model.fit(X_train,y_train)

								
								preds = cv_model.best_estimator_.predict(X_test)

								acc = accuracy_score(y_test, preds)
								b_acc = balanced_accuracy_score(y_test,preds)
								results['acc'].append(acc)
								results['bal_acc'].append(b_acc)
								
								if model == 'LR':
									prob_preds = cv_model.best_estimator_.predict_proba(X_test)
									roc_auc = roc_auc_score(y_test, prob_preds[:,1])
									prec, rec, thresh = precision_recall_curve(y_test,prob_preds[:,1])
									pr_auc = auc(rec,prec)
									results['roc_auc'].append(roc_auc)
									results['pr_auc'].append(pr_auc)
								else:
									results['roc_auc'].append(-1)
									results['pr_auc'].append(-1) 
							


	results = pd.DataFrame(results)

	results.to_csv(path + "DW.csv", index = False)
		
if __name__ == '__main__':
	main()



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


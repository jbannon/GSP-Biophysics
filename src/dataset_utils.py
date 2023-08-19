from collections import defaultdict

from rdkit import DataStructs


from typing import List, Tuple, Dict, NamedTuple

import pandas as pd
import numpy as np
import sys 


from ogb.utils import mol

from rdkit.Chem import Draw
import networkx as nx
from rdkit import Chem
import os 
import pickle 
from rdkit.Chem import AllChem
import os

import GraphTransforms as gt 

FINGERPRINT_FEATURES = ['morgan_short','morgan_long','top_short','top_long','bit_short','bit_long']

GSP_FEATURES = ["DW","DWLG"]


REGRESSION_DATASETS = ['HydrationFreeEnergy_FreeSolv','Lipophilicity_AstraZeneca']

CLASSIFICATION_DATASETS = ['Bioavailability_Ma','PAMPA_NCATS','PAMPA_APPROVED','HIA_Hou','BBB_Martins']

dataset_dict = {'regression':['HydrationFreeEnergy_FreeSolv','Lipophilicity_AstraZeneca'],
				'classification':['Bioavailability_Ma','PAMPA_NCATS','PAMPA_APPROVED','HIA_Hou','BBB_Martins']}


dataset_to_short_name = {'HydrationFreeEnergy_FreeSolv': 'HFE', 
						 'Lipophilicity_AstraZeneca': 'Lipophilicity',
						 'PAMPA_NCATS': 'MemPerm', 
						 'BBB_Martins': 'BBB',
						 'Bioavailability_Ma': 'Bioavail',
						 'PAMPA_APPROVED': 'MemPerm_Approved',
						 'HIA_Hou': 'HIA',
						 'BBB_Martins': 'BBB'
						}


def reset_dict_index(dataset:Dict) -> Dict:
	keymap = {}
	
	old_keys = list(dataset.keys())
	
	for i in range(len(old_keys)):
		keymap[i] = old_keys[i]


	new_ds = {i:dataset[keymap[i]] for i in keymap.keys()}
	
	return new_ds



def make_linegraph(
	graph,
	adj_mat
	) -> Dict:
	

	
	L = nx.line_graph(nx.from_numpy_array(adj_mat))
	
	LG_node_feats = defaultdict(list)

	edge_list = graph['edge_index']
	edge_feat = graph['edge_feat']
	

	for i in range(edge_list.shape[1]):
		node1, node2 = edge_list[0,i], edge_list[1,i]
		features = edge_feat[i,:]
		edge = (min(node1,node2),max(node1,node2))

		LG_node_feats[edge].append(features)
					
	for k in LG_node_feats.keys():
		efeat = LG_node_feats[k]
		if len(efeat) !=2:
			print("issue with edge {e}".format(e=k))
		if not (efeat[0]==efeat[1]).all():
			print("issue with edge {e}".format(e=k))
			sys.exit(1)
		else:
			LG_node_feats[k] = efeat[0]
	
	A = nx.adjacency_matrix(L).todense()
	
	node_feat = np.array([])
	for node in L.nodes():
		node_feat = np.vstack((node_feat,LG_node_feats[node])) if node_feat.size else LG_node_feats[node]
	
	return A, node_feat




def make_adj_mat(
	g
	) -> np.ndarray:
	
	N = g['num_nodes']
	adj_mat = np.zeros((N,N))
	
	for i in range(g['edge_index'].shape[1]):
		n1, n2 = g['edge_index'][0,i], g['edge_index'][1,i]
		adj_mat[n1,n2] = 1
	if not (adj_mat == adj_mat.T).all():
		sys.exit(1)
	return adj_mat


def make_dataset(
	df:pd.DataFrame,
	short_size:int = 32,
	long_size:int = 1024
	) -> None:	

	
	dataset = {}


	bit_short_gen = AllChem.GetRDKitFPGenerator(fpSize=short_size)
	bit_long_gen = AllChem.GetRDKitFPGenerator(fpSize = long_size)

	top_short_gen = AllChem.GetTopologicalTorsionGenerator(fpSize = short_size)
	top_long_gen = AllChem.GetTopologicalTorsionGenerator(fpSize = long_size)

	morgan_short_gen = AllChem.GetMorganGenerator(fpSize = short_size )
	morgan_long_gen = AllChem.GetMorganGenerator(fpSize = long_size)


	for idx, row in df.iterrows():
		

		mol_graph = mol.smiles2graph(row['Drug'])
		if mol_graph['num_nodes'] <= 4:
			continue
		molecule = Chem.MolFromSmiles(row['Drug'])
		
		adjacency_matrix = make_adj_mat(mol_graph)
		
		bit_short, bit_long, top_short, top_long, morgan_short, morgan_long = [np.array([]) for i in range(6)]
		
		DataStructs.ConvertToNumpyArray(bit_short_gen.GetFingerprint(molecule),bit_short)
		DataStructs.ConvertToNumpyArray(bit_long_gen.GetFingerprint(molecule),bit_long)

		DataStructs.ConvertToNumpyArray(top_short_gen.GetFingerprint(molecule),top_short)
		DataStructs.ConvertToNumpyArray(top_long_gen.GetFingerprint(molecule),top_long)

		DataStructs.ConvertToNumpyArray(morgan_short_gen.GetFingerprint(molecule),morgan_short)
		DataStructs.ConvertToNumpyArray(morgan_long_gen.GetFingerprint(molecule),morgan_long)

		
		

		lg_adj_mat, lg_node_feat = make_linegraph(mol_graph,adjacency_matrix)
		
		
		

		dataset[idx] = {
			'node_feat': mol_graph['node_feat'],
			'num_nodes': mol_graph['num_nodes'],
			'adj_mat':adjacency_matrix,
			'bit_short': bit_short,
			'bit_long': bit_long,
			'top_short': top_short,
			'top_long': top_long,
			'morgan_short': morgan_short,
			'morgan_long': morgan_long,
			'edge_feat': mol_graph['edge_feat'],
			'linegraph': {'adj_mat':lg_adj_mat,
						   'node_feat': lg_node_feat
						  },
			'y':row['Y']
			}

	dataset = reset_dict_index(dataset)
	return dataset


def make_numpy_dataset(
	data:Dict,
	feature_type:str,
	numScales_v:int = None,
	maxMoment_v:int = None,
	central_v:bool = True,

	numScales_e:int = None,
	maxMoment_e: int = None,
	central_e:bool = True
	):
	

	X, y = [np.array([]) for i in range(2)]

	if feature_type in FINGERPRINT_FEATURES:
		for k in data.keys():
			molecule_data = data[k]
			
			X = np.vstack( (X, molecule_data[feature_type])) if X.size else molecule_data[feature_type]
			y = np.vstack( (y, molecule_data['y'])) if  y.size else np.array(molecule_data['y'])
	elif feature_type.upper() == 'DW':
		for k in data.keys():
			molecule_data = data[k]
			

			transformer = gt.DiffusionWMT(numScales_v,maxMoment_v,molecule_data['adj_mat'],central_v)
			X_transformed = transformer.computeTransform(molecule_data['node_feat'])
			
			X = np.vstack( (X, X_transformed)) if X.size else X_transformed
			y = np.vstack( (y, molecule_data['y'])) if  y.size else np.array(molecule_data['y'])

	elif feature_type.upper() == "DWLG":
		for k in data.keys():
			molecule_data = data[k]

		

			transformer = gt.DiffusionWMT(numScales_v,maxMoment_v,molecule_data['adj_mat'],central_v)
			LG_transform = gt.DiffusionWMT(numScales_e,maxMoment_e,molecule_data['linegraph']['adj_mat'],central_e)
			

			X_transformed = transformer.computeTransform(molecule_data['node_feat'])
			
			X_LG_transformed = LG_transform.computeTransform(molecule_data['linegraph']['node_feat'])

			X_transformed = np.hstack((X_transformed,X_LG_transformed))

		
			
			
			X = np.vstack( (X, X_transformed)) if X.size else X_transformed
			y = np.vstack( (y, molecule_data['y'])) if  y.size else np.array(molecule_data['y'])


	return X, y

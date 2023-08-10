from collections import defaultdict
import pubchempy as pcp 
from rdkit import DataStructs
from typing import List, Tuple, Dict, NamedTuple
from typing import Union, List, Dict, Tuple
import pandas as pd
import numpy as np
import sys 
from ogb.utils import mol
from tdc.single_pred import ADME
from rdkit.Chem import Draw
import networkx as nx
from rdkit import Chem
import os 
import pickle 
from rdkit.Chem import AllChem
import os

Dataset_Names = {
	"Hydration_Free_Energy":'HydrationFreeEnergy_FreeSolv',
	"Lipophilicity":'Lipophilicity_AstraZeneca',
	'Membrane_Permeability':"PAMPA_NCATS"
	}



def make_diffusion_op(
	W:np.ndarray
	) -> np.ndarray:
	D = np.diag(np.sum(W,axis=1))
	D_sqrt = np.sqrt(D)
	A = D_sqrt @ W @ D_sqrt
	T = 0.5*(np.eye(A.shape[0])+A)
	return(T)
	







def make_pmatrix(
	M:np.ndarray
	) -> None:
	
	lines = str(M).replace('[', '').replace(']', '').replace('.','').splitlines()
	rv = [r'\begin{pmatrix}']
	rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
	rv +=  [r'\end{pmatrix}']
	return '\n'.join(rv)



def make_fig(
	molecule,
	adjacency_matrix
	) -> None:
	pm = make_pmatrix(adjacency_matrix)
	Draw.MolToFile(molecule,"../figs/examples/molecule.png")
	with open("../figs/examples/adjacency_matrix.tex","w") as f:
		f.writelines(pm)
	





def make_dataset(
	df:pd.DataFrame,
	madeFig: bool
	) -> None:	

	
	dataset = {}
	for idx, row in df.iterrows():
		

		mol_graph = mol.smiles2graph(row['Drug'])
		if mol_graph['num_nodes'] <= 4:
			continue
		molecule = Chem.MolFromSmiles(row['Drug'])
		adjacency_matrix = make_adj_mat(mol_graph)
		T = make_diffusion_op(adjacency_matrix)
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
			'diff_op':T,
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

		if mol_graph['num_nodes']<= 10 and mol_graph['num_nodes'] > 5 and not madeFig:
			make_fig(molecule,adjacency_matrix)
			madeFig = True
	dataset = reset_dict_index(dataset)
	return dataset, madeFig





path = "../data/raw/tdc/"

os.makedirs("../figs/examples/",exist_ok = True)
os.makedirs("../data/processed/",exist_ok = True)



madeFig = False

for dsname in Dataset_Names.keys():
	print(dsname)
	os.makedirs("../data/processed/{ds}".format(ds=dsname),exist_ok = True)
	if dsname == 'Membrane_Permeability':
		altname = 'MemPerm_Approved'
		os.makedirs("../data/processed/{ds}".format(ds=altname),exist_ok = True)

	fname = Dataset_Names[dsname]
	data = ADME(name = fname, path = path)
	df = data.get_data()
	
	dataset, madeFig = make_dataset(df,madeFig)

	with open("../data/processed/{ds}/processed.pickle".format(ds=dsname),'wb') as f:
		pickle.dump(dataset,f)

	if dsname == 'Membrane_Permeability':
		df = data.get_approved_set()
		dataset, madeFig = make_dataset(df,madeFig)

		with open("../data/processed/{ds}/processed.pickle".format(ds=altname),'wb') as f:
			pickle.dump(dataset,f)

		
		
		
		
	

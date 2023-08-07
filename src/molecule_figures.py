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



def make_pmatrix(
	M:np.ndarray
	) -> None:
	M = M.astype(int)
	lines = str(M).replace('[', '').replace(']', '').replace(".",'').splitlines()
	print(lines)
	sys.exit(1)
	rv = [r'\begin{pmatrix}']
	rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
	rv +=  [r'\end{pmatrix}']
	return '\n'.join(rv)


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


compound_name = 'Erlotinib'
results = pcp.get_compounds(compound_name, 'name')
smiles = results[0].isomeric_smiles
molecule = Chem.MolFromSmiles(smiles)
graph = mol.smiles2graph(smiles)
AM = make_adj_mat(graph)

pm = make_pmatrix(AM)
with open("figs/erlotinib.tex","w") as f:
	f.writelines(pm)



# print(str(AM))
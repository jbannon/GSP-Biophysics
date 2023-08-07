import sys
from sklearn.model_selection import LeaveOneOut,GridSearchCV, KFold, StratifiedKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from pytorch_forecasting.metrics import QuantileLoss

from torch_geometric.datasets import TUDataset

import pandas as pd
import torch

from ogb.utils import mol
from tdc.single_pred import ADME


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
	def __init__(self, 
		hidden_channels:int,
		node_feat_dim:int,
		num_quantiles:int):
		super(GCN, self).__init__()
		
		torch.manual_seed(12345)
		self.conv1 = GCNConv(node_feat_dim, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.conv3 = GCNConv(hidden_channels, hidden_channels)
		self.lin = Linear(hidden_channels, num_quantiles)

	def forward(self, x, edge_index, batch):

		# 1. Obtain node embeddings 
		x = self.conv1(x, edge_index)
		x = x.relu()
	
		x = self.conv2(x, edge_index)
		x = x.relu()
		x = self.conv3(x, edge_index)

		# 2. Readout layer
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

		# 3. Apply a final classifier
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin(x)

		return x


dataset = TUDataset(root='data/TUDataset', name='MUTAG')







data = ADME(name = 'Lipophilicity_AstraZeneca', path = "../data/raw/tdc/")

df = data.get_data()
# print(df.head())

adme_data = []

for idx, row in df.iterrows():		
	mol_graph = mol.smiles2graph(row['Drug'])
	mol_graph_ = Data(x = torch.from_numpy(mol_graph['node_feat']).float(),
		edge_index = torch.from_numpy(mol_graph['edge_index']),
		edge_attr = torch.from_numpy(mol_graph['edge_feat']).float(),
		y = torch.Tensor([row['Y']]).float())
	
	
	

	adme_data.append(mol_graph_)


def quantile_loss(output, target):
	alpha = 0.95
	error = target-output
	return torch.maximum(alpha*error,(1-alpha)*error)



    

for i in range(10):
	train_idx, test_idx = train_test_split(range(len(adme_data)))
	train_dataset = [adme_data[i] for i in train_idx]
	test_dataset = [adme_data[i] for i in test_idx]

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
	
	print(train_dataset[0])
	
	model = GCN(hidden_channels=64,node_feat_dim = 9, num_quantiles =1)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	criterion = quantile_loss()
	

	model.train()

	for data in train_loader:  # Iterate in batches over the training dataset.
		out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
		loss = criterion(out, data.y)  # Compute the loss.
		loss.backward()  # Derive gradients.
		optimizer.step()  # Update parameters based on gradients.
		optimizer.zero_grad()  # Clear gradients.

	model.eval()

	correct = 0
	for data in test_loader:  # Iterate in batches over the training/test dataset.
		out = model(data.x, data.edge_index, data.batch)  
		print(float(criterion(out,data.y)))
		sys.exit(0)






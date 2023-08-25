from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import umap 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from tdc.single_pred import ADME
import dataset_utils
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import os


model = Pipeline([('scaler',StandardScaler()),('dimred',PCA(n_components = 2))])


for name in ['Bioavailability_Ma','PAMPA_NCATS','HIA_Hou','BBB_Martins']:
	data = ADME(name = name,path = "../data/raw/tdc")
	p = "../figs/embeddings/{d}".format(d=name)
	os.makedirs(p,exist_ok=True)
	for feature_type in ['DW','DWLG','morgan-short']:
		dataframe = data.get_data()
		DS = dataset_utils.make_dataset(dataframe,32,100)
		X, y = dataset_utils.make_numpy_dataset(DS, feature_type = feature_type, numScales_v = 2, maxMoment_v = 3, central_v = False, 
			numScales_e = 2, maxMoment_e = 3, central_e = False)

		X = StandardScaler().fit_transform(X)

		u = umap.UMAP()
		emb1 = u.fit_transform(X)

		plt.scatter(emb1[:,0], emb1[:,1],c= y)
		plt.savefig("{path}/umap_{ft}.png".format(path = p,ft = feature_type ))
		plt.close()
		emb2 = TSNE(n_components=2, learning_rate='auto',
			init='random', perplexity=3).fit_transform(X)

		plt.scatter(emb2[:,0], emb2[:,1],c= y)
		plt.savefig("{path}/tsne_{ft}.png".format(path = p,ft = feature_type ))
		plt.close()
		

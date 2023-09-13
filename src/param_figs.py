import sys
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import os





path = "../figs/sensitivity/classification/"
os.makedirs(path, exist_ok=True)



transform_dict = {'DW':['approved_DW.csv','full_DW.csv'],'DWLG': ['approved_DWLG.csv','full_DWLG.csv']}


folder = "../results/sensitivity/classification/"
for transform in transform_dict.keys():
	files = transform_dict[transform]
	
	for file in files:
		df = pd.read_csv(folder + file)
		ds = 'full' if 'full' in file else 'approved'

		for m in pd.unique(df['model']):
			if m == 'LR':
				metrics = ['acc','bal_acc','roc_auc','pr_auc']
			else:
				metrics = ['acc','bal_acc']
			for cent in pd.unique(df['centered_vertex_features']):
				
				sub = df[(df['model']==m)  & (df['centered_vertex_features']==cent)]
			
				for metric in metrics:
					values = sub[['max_vertex_scale','max_vertex_moment',metric]]

					tab = pd.crosstab(index = values['max_vertex_scale'],
						columns = values['max_vertex_moment'],
						values = values[metric], aggfunc = 'mean')
					ax = sns.heatmap(tab)
					
					ax.set(title = "{m} performance in on {ds} dataset \n in terms of {e} vs. parameter".format(m=m,ds =ds, e=metric),
					xlabel = "Maximum Vertex Moment",
					ylabel = "Maximum Vertex_Scale")
					# plt.show()
					# sys.exit(1)
					plt.savefig(path+"{m}_{e}_{ds}.png".format(m=m,e=metric,ds = ds))
					plt.close()


			
# df = pd.read_csv("../results/sensitivity/regression/DW.csv")


# print(df.columns)
# path = "../figs/sensitivity/regression/"
# os.makedirs(path, exist_ok=True)



# metrics = ['MAE','MSE']
# for pca in ['Yes','No']:
# 	for cent in pd.unique(df['centered_vertex_features']):
			
# 		sub = df[(df['pca']==pca) & (df['centered_vertex_features']==cent)]
	
# 		for metric in metrics:
# 			values = sub[['max_vertex_scale','max_vertex_moment',metric]]

# 			tab = pd.crosstab(index = values['max_vertex_scale'],
# 				columns = values['max_vertex_moment'],
# 				values = values[metric], aggfunc = 'mean')
# 			ax = sns.heatmap(tab)
			
# 			ax.set(title = "Ridge performance in terms of {e}\n vs. parameter".format(m=m,e=metric),
# 			xlabel = "Maximum Vertex Moment",
# 			ylabel = "Maximum Vertex_Scale")

# 			plt.savefig(path+"{e}_{p}_{c}.png".format(m=m,e=metric,p=pca,c=cent))
# 			plt.close()


			

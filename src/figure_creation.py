import sys
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import os 

# sns.set_theme(style="ticks", palette="pastel")


# for feat in feats:
# 	new_df = pd.read_csv("../results/paucity/{ds}/{m}/{f}.csv".format(ds=dataset,m=model,f=feat))
# 	df = pd.concat([df,new_df],axis=0)


# print(df.head())
# print(df.shape)


# sns.boxplot(x="test_pct", y="acc",
#             hue="feature",
#             data=df)

# plt.savefig("../figs/bioavail_paucity_SVC.png")
# plt.close()


	for model in ['RFR','RIDGE']:
		df = pd.DataFrame()
		os.makedirs("../figs/regression/{ds}/".format(ds=dataset),exist_ok = True)
		for feat in ['morgan_short','morgan_long','DW','DWLG']:
			if feat in ['DW','DWLG'] and model == 'RFR' and dataset == 'Lipophilicity':
				continue
			new_df = pd.read_csv("../results/regression/{ds}/{m}/{f}.csv".format(ds =dataset, m = model, f = feat))
			df = pd.concat([df,new_df],axis=0)

		for measure in ['MAE','RMSE']:
			plot_order = df.groupby('feature')[measure].mean().sort_values(ascending=True).index.values
			print(plot_order)
			
			sns.boxplot(x = "feature", y = measure, hue = 'feature',data = df, order=plot_order).set(
				title = "Performance of {m} on {d} Dataset\n vs. Molecule Representation".format(m=model, d = dataset ),
				xlabel = 'Feature Transform')
			plt.legend([],[],frameon=False)

			plt.savefig("../figs/regression/{ds}/{m}_{e}.png".format(ds = dataset,m = model, e = measure))
			plt.close()
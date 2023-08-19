import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 


sns.set_theme(style="ticks", palette="pastel")

dataset = "Bioavail"
model = "SVC"
feats = ['morgan_short','DW','DWLG']
df = pd.DataFrame()

for feat in feats:
	new_df = pd.read_csv("../results/paucity/{ds}/{m}/{f}.csv".format(ds=dataset,m=model,f=feat))
	df = pd.concat([df,new_df],axis=0)


print(df.head())
print(df.shape)


sns.boxplot(x="test_pct", y="acc",
            hue="feature",
            data=df)

plt.savefig("../figs/bioavail_paucity_SVC.png")
plt.close()

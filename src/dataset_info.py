import pandas as pd 
from tdc.single_pred import ADME



print("\n")

for name in ['Bioavailability_Ma','PAMPA_NCATS','HIA_Hou','BBB_Martins']:

	data = ADME(name = name,path = "../data/raw/tdc")

	df = data.get_data()
	print(name)
	print(df['Y'].value_counts())
	
	if name == 'PAMPA_NCATS':
		print("{n} approved".format(n=name))
		df = data.get_approved_set()
		print(df['Y'].value_counts())
	print("\n")


	
	
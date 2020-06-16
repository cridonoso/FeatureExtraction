import FATS
import os, sys
import multiprocessing as mp
import numpy as np
import h5py
import pandas as pd
try:
	import util_records as data
	import tensorflow as tf
except:
	print('No tensorflow')
import time
from features_name import most_important, harmonics, others

class_code = {'wise':{'NC':0, 'RRab':1, 'RRc':2, 'DSCT_SXPHE':3, 'CEP':4, 
			  		  'SRV':5, 'Mira':6, 'OSARG':7, 'NonVar':8},
			  'ogle':{'cep': 0, 'RRab': 1, 'RRc': 2, 'dsct': 3, 'EC': 4, 'ED': 5, 
			  		  'ESD': 6, 'Mira': 7, 'SRV': 8, 'OSARG': 9, 'std': 10},
			  'gaia':{'CEP': 0, 'T2CEP': 1, 'MIRA_SR': 2, 'DSCT_SXPHE':3, 'RRAB':4, 'RRC':5, 'RRD':6},
			  'css':{'ACEP':0, 'Blazkho':1, 'CEPII':2, 'DSC':3, 'EA':4, 'EA_UP':5, 'ELL':6,
		       		  'EW':7, 'HADS':8, 'LPV':9, 'Misc':10, 'RRab':11, 'RRc':12,
		       		  'RRd':13, 'RS_CVn':14, 'Rotational Var':15, 'Transient':16, 'beta_Lyrae':17},
			  'macho': {'QSO':0, 'Be':1, 'CEPH':2, 'RRL':3, 'EB':4, 'MOA':5, 'LPV':6},
			  'linear': {'RRLab':0, 'RRLc':1, 'Eclipsing Algol':2, 'Contact binary':3, 'DSCT':4},
			  'asas': {'Beta Persei':0, 'Classical Cepheid':1, 'RR Lyrae FM':2, 'Semireg PV':3, 'W Ursae Ma':4}
}

skip = {'wise': ['ACEP', 'ARRD', 'C', 'ELL', 'T2CEP', 'RRd'],
		'ogle': [],
		'gaia': ['ACEP', 'ARRD'],
		'css': ['CEPI', 'LADS', 'PCEB', 'Hump'],
		'macho': [],
		'linear': [],
		'asas':[]}

col_names = {'wise':[], 
			 'ogle':['mjd', 'mag', 'errmag', 'a', 'b', 'c'], 
			 'gaia':[],
			 'macho':[],
			 'linear':[],
			 'css':[],
			 'asas':[]}

delim_whitespaces = {'wise': False, 
				  	 'ogle':True, 
				  	 'gaia':False, 
				  	 'macho':False,
				  	 'linear':True,
				  	 'css':False,
				  	 'asas':False}



def process(path_lcs, lc_id, names, delim_whitespace):
	if names == []:
		df = pd.read_csv('{}/{}'.format(path_lcs, lc_id), 
										delim_whitespace=delim_whitespace)
	else:
		df = pd.read_csv('{}/{}'.format(path_lcs, lc_id), 
							delim_whitespace=delim_whitespace,
							names=names)
	return [df.iloc[:,:3].min().values, df.iloc[:,:3].max().values]

def get_moments(dataframe, path_lcs, names=[], delim_whitespace=True):
	print('[INFO] Finding min and max values all class objects')
	

	num_cores = mp.cpu_count()
	pool = mp.Pool(processes=num_cores)
	results = []
	for k, row in dataframe.iterrows():
		lc_info = row['Path'].split('/')
		results.append(pool.apply_async(process, args=(path_lcs,
													   lc_info[-1], 
													   names, 
													   delim_whitespace)))
		# if k == 5:break
	values = np.array([p.get() for p in results])

	min_values = np.min(values[:,0,:], 0)
	max_values = np.max(values[:,1,:], 0)
	return {'min': min_values, 'max':max_values}

def run_fats(lc, label):
	fs = FATS.FeatureSpace(Data=['magnitude','time', 'error'], featureList=most_important+harmonics+others)
	results = fs.calculateFeature([lc[:,1], lc[:,0], lc[:,2]])	

	dic = results.result(method='dict')
	
	array = np.array(list(dic.values()), dtype=np.float32)
	array = np.nan_to_num(np.concatenate([array, label]))

	return array


def rf_features_from_dat(path_meta, path_lcs, path_to_save, name,  norm='n1'):
	os.makedirs(path_to_save, exist_ok = True)

	metadata_df = pd.read_csv(path_meta)
	metadata_df = metadata_df[~metadata_df.Class.isin(skip[name])]
	metadata_df = metadata_df[metadata_df['N'] >=10]
	n_classes = len(class_code)

	if norm == 'n1':
		min_max_by_class = get_moments(metadata_df, path_lcs, 
						   names=col_names[name], 
						   delim_whitespace=delim_whitespaces[name])

	df_train = metadata_df.sample(frac=0.5)
	df_test  = metadata_df.loc[~metadata_df.index.isin(df_train.index)]

	print(len(np.unique(df_train.Class)))
	print(len(np.unique(df_test.Class)))

	df_test.to_csv(path_to_save+'/test_curves.csv')

	num_cores = mp.cpu_count()

	for dataframe, dsname in zip([df_train, df_test], ['train', 'test']):
		pool = mp.Pool(processes=num_cores)
		results = []
		starttime = time.time()
		count = 0 
		for k, row in dataframe.iterrows():
			lc_info = row['Path'].split('/')

			if col_names[name] == []:
				df = pd.read_csv('{}/{}'.format(path_lcs, lc_info[-1]), 
												delim_whitespace=delim_whitespaces[name])
			else:
				df = pd.read_csv('{}/{}'.format(path_lcs, lc_info[-1]), 
									delim_whitespace=delim_whitespaces[name],
									names=col_names[name])

			df = df.iloc[:,:3]
			
			if norm == 'n1':
				min_v = min_max_by_class['min']
				max_v = min_max_by_class['max']
				normalized_df = (df-min_v)/(max_v - min_v)

			if norm == 'n2':
				mean_ = df.mean()
				std_ = df.std()
				normalized_df = (df-mean_)/(std_)
	
			normalized_df = np.nan_to_num(normalized_df.values)

			

			results.append(pool.apply_async(run_fats, args=(normalized_df, [class_code[name][row['Class']]])))

			# if count == 5: break
			# count+=1


		features = np.array([p.get() for p in results])
		elapsed = time.time() - starttime
		print ('FATS total run time: ',elapsed,'seg for ', features.shape[0],'samples')
		# Writting g5 file with features
		with h5py.File('{}/{}_features.h5'.format(path_to_save, dsname), 'w') as hf:
			hf.create_dataset('features', data=features[..., :-1])
			hf.create_dataset('labels', data=features[..., -1])

def online_features_from_dat(path, path_lcs, path_to_save, name, tokens=[], norm='n1'):
	if tokens == []:
		tokens = np.arange(10, 210, 10)
	
	os.makedirs(path_to_save, exist_ok=True)

	metadata_df = pd.read_csv(path)
	n_classes = len(class_code)

	if norm == 'n1':
		min_max_by_class = get_moments(metadata_df, path_lcs, 
						   names=col_names[name], 
						   delim_whitespace=delim_whitespaces[name])

	num_cores = mp.cpu_count()
	print('[INFO] Online computation')
	print ('[INFO] Using', num_cores,'cores')
	times = []
	 
	with h5py.File('{}/online_features.h5'.format(path_to_save), 'w') as hf:
		for lim in tokens:
			starttime = time.time()
			pool = mp.Pool(processes=num_cores)
			results = []
			count = 0
			for k, row in metadata_df.iterrows():
				lc_info = row['Path'].split('/')
				if row['N'] < lim: continue
				if col_names[name] == []:
					df = pd.read_csv('{}/{}'.format(path_lcs, lc_info[-1]), 
													delim_whitespace=delim_whitespaces[name])
				else:
					df = pd.read_csv('{}/{}'.format(path_lcs, lc_info[-1]), 
													delim_whitespace=delim_whitespaces[name],
													names=col_names[name])

				df = df.iloc[0:lim,:3]

				if norm == 'n1':
					min_v = min_max_by_class['min']
					max_v = min_max_by_class['max']
					normalized_df = (df-min_v)/(max_v - min_v)

				if norm == 'n2':
					mean_ = df.mean()
					std_ = df.std()
					normalized_df = (df-mean_)/(std_)
					
				normalized_df = np.nan_to_num(normalized_df.values)
				results.append(pool.apply_async(run_fats, args=(normalized_df, [class_code[name][row['Class']]])))
				
				# if count==10:break
				# count+=1


			features = np.array([p.get() for p in results])
			elapsed = time.time() - starttime

			times.append(elapsed)	
			hf.create_dataset(str(lim), data=features)
			print('[INFO] {} obs done! {} seconds for {} curves'.format(lim, elapsed, features.shape[0]))
		hf.create_dataset('time', data=np.array(times))



if __name__ == '__main__':
	name = sys.argv[1] # /ogle/test_0
	main_path = sys.argv[2] #'../datasets/raw_data/'
	norm = sys.argv[3]
	path_meta = '{}/{}/{}/{}_dataset.dat'.format(main_path, name, name.upper(), name.upper())
	path_lcs  = '{}/{}/{}/LCs/'.format(main_path, name, name.upper())
	path_to_save = '/home/shared/cridonoso/datasets/features/{}'.format(name)
	
	rf_features_from_dat(path_meta, path_lcs, path_to_save, name, norm=norm)

	path_test_meta = '/home/shared/cridonoso/datasets/{}/test_curves.csv'.format(name)

	online_features_from_dat(path_test_meta, path_lcs, path_to_save, name, norm=norm)

	
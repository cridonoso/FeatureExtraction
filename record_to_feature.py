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

class_code = {'wise':{'NC':0, 'RRab':1, 'RRc':2, 'DSCT_SXPHE':3, 'CEP':4, 
			  		  'SRV':5, 'Mira':6, 'OSARG':7, 'NonVar':8},
			  'ogle':{'cep': 0, 'RRab': 1, 'RRc': 2, 'dsct': 3, 'EC': 4, 'ED': 5, 
			  		  'ESD': 6, 'Mira': 7, 'SRV': 8, 'OSARG': 9, 'std': 10},
			  'gaia':{'CEP': 0, 'T2CEP': 1, 'MIRA_SR': 2, 'DSCT_SXPHE':3, 
			  		  'RRAB':4, 'RRC':5, 'RRD':6},
			  'css':{'ACEP':1, 'Blazkho':2, 'CEPII':3, 'DSC':4, 'EA':5, 'EA_UP':6, 'ELL':7,
		       		  'EW':8, 'HADS':9, 'LPV':10, 'Misc':11, 'RRab':12, 'RRc':13,
		       		  'RRd':14, 'RS_CVn':15, 'Rotational Var':15, 'Transient':16, 'beta_Lyrae':17},
			  'macho': {'QSO':0, 'Be':1, 'CEPH':2, 'RRL':3, 'EB':4, 'MOA':5, 'LPV':6},
			  'linear': {'RRLab':0, 'RRLc':1, 'Eclipsing Algol':2, #eclipsing algol
			  			 'Contact binary':3, 'DSCT':4},
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

most_important = ['MedianAbsDev',
				 'PeriodLS',
				 'Autocor_length',
				 'Q31', 
				 'FluxPercentileRatioMid35',
				 'FluxPercentileRatioMid50',
				 'FluxPercentileRatioMid65',
				 'FluxPercentileRatioMid20',
				 'Beyond1Std',
			     'Skew', 
				 'Con'
				 'Gskew',
				 'Meanvariance', 
				 'StetsonK',
				 'FluxPercentileRatioMid80',
				 'CAR_sigma',
				 'CAR_mean', 
				 'CAR_tau',
				 'MedianBRP',
				 'Rcs',
			     'Std',
				 'SmallKurtosis',
				 'Mean', 
				 'Amplitude',
				 'StructureFunction_index_21',
				 'StructureFunction_index_31'] 

harmonics = ['Freq1_harmonics_amplitude_0',
		     'Freq1_harmonics_amplitude_1',
			 'Freq1_harmonics_amplitude_2',
			 'Freq1_harmonics_amplitude_3',
			 'Freq1_harmonics_rel_phase_0',
			 'Freq1_harmonics_rel_phase_1',
			 'Freq1_harmonics_rel_phase_2',
			 'Freq1_harmonics_rel_phase_3',
			 'Freq2_harmonics_amplitude_0',
			 'Freq2_harmonics_amplitude_1',
			 'Freq2_harmonics_amplitude_2',
			 'Freq2_harmonics_amplitude_3',
			 'Freq2_harmonics_rel_phase_0',
			 'Freq2_harmonics_rel_phase_1',
			 'Freq2_harmonics_rel_phase_2',
			 'Freq2_harmonics_rel_phase_3',
			 'Freq3_harmonics_amplitude_0',
			 'Freq3_harmonics_amplitude_1',
			 'Freq3_harmonics_amplitude_2',
			 'Freq3_harmonics_amplitude_3',
			 'Freq3_harmonics_rel_phase_0',
			 'Freq3_harmonics_rel_phase_1',
			 'Freq3_harmonics_rel_phase_2',
			 'Freq3_harmonics_rel_phase_3']

others = ['Psi_CS', 'Psi_eta', 'AndersonDarling', 
		  'LinearTrend', 'PercentAmplitude', 
		  'MaxSlope', 'PairSlopeTrend', 'Period_fit', 
		  'StructureFunction_index_32']

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
	results = fs.calculateFeature([lc[:,0], lc[:,1], lc[:,2]])	

	dic = results.result(method='dict')
	
	array = np.array(list(dic.values()), dtype=np.float32)
	array = np.nan_to_num(np.concatenate([array, label]))

	return array


def calculate_features(path, name, n_samples=-1, multiprocessing=False):
		
	path_to_save = './features/{}'.format(name)
	os.makedirs(path_to_save, exist_ok=True)

	light_curves = data.load_record(path, 1, n_samples=n_samples)

	starttime = time.time()
	

	if multiprocessing:
		processes = []
		num_cores = mp.cpu_count()
		print ('[INFO] Using',num_cores,'cores')
		pool = mp.Pool(processes=num_cores)
	
		results = []
		for x, y, m in light_curves:
			lc = tf.boolean_mask(x[0], m[0]).numpy()
			if lc.shape[0] >= 10:
				results.append(pool.apply_async(run_fats, args=(lc, y.numpy())))
		features = np.array([p.get() for p in results])
	else:
		print('[INFO] Using 1 core')
		results = []
		for x, y, m in light_curves:
			lc = tf.boolean_mask(x[0], m[0]).numpy()
			if lc.shape[0] >= 10:
				results.append(run_fats(lc, y.numpy()))		
		features = np.array(results, dtype=np.float32)
	
	elapsed = time.time() - starttime
	print ('FATS total run time: ',elapsed,'seg for ', features.shape[0],'samples')
	# Writting g5 file with features
	with h5py.File('{}/features.h5'.format(path_to_save), 'w') as hf:
		hf.create_dataset('features', data=features)


def calculate_online_features(path, path_to_save, tokens=[], n_samples=-1, multiprocessing=False):
	if tokens == []:
		tokens = np.arange(10, 210, 10)
	
	
	os.makedirs(path_to_save, exist_ok=True)

	light_curves = data.load_record(path, 1, n_samples=n_samples)

	num_cores = mp.cpu_count()
	print('[INFO] Online computation')
	print ('[INFO] Using', num_cores,'cores')
	times = []
	with h5py.File('{}/online_features.h5'.format(path_to_save), 'w') as hf:
		for lim in tokens:
			if multiprocessing:
				starttime = time.time()
				pool = mp.Pool(processes=num_cores)
				results = []
				for i, (x, y, m )in enumerate(light_curves):
					lc = tf.boolean_mask(x[0], m[0]).numpy()[:lim]
					if lc.shape[0] >= 10:
						results.append(pool.apply_async(run_fats, args=(lc, y.numpy())))
				features = np.array([p.get() for p in results])
				elapsed = time.time() - starttime
			else:
				starttime = time.time()
				for i, (x, y, m )in enumerate(light_curves):
					print('Processing curve {}'.format(i), end='\n')
					lc = tf.boolean_mask(x[0], m[0]).numpy()[:lim]
					if lc.shape[0] >= 10:
						results.append(run_fats(lc, y.numpy()))
				features = np.array(results, dtype=np.float32)
				elapsed = time.time() - starttime

			times.append(elapsed)	
			hf.create_dataset(str(lim), data=features)
			print('[INFO] {} obs done! {} seconds for {} curves'.format(lim, elapsed, features.shape[0]))
		hf.create_dataset('time', data=np.array(times))

def rf_features_from_dat(path_meta, path_lcs, path_to_save, name):
	os.makedirs(path_to_save, exist_ok = True)

	metadata_df = pd.read_csv(path_meta)
	metadata_df = metadata_df[~metadata_df.Class.isin(skip[name])]
	metadata_df = metadata_df[metadata_df['N'] >=10]
	n_classes = len(class_code)

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
			
			min_v = min_max_by_class['min']
			max_v = min_max_by_class['max']
			normalized_df = (df-min_v)/(max_v-min_v)

			

			results.append(pool.apply_async(run_fats, args=(normalized_df.values, [class_code[name][row['Class']]])))

			# if count == 5: break
			# count+=1


		features = np.array([p.get() for p in results])
		elapsed = time.time() - starttime
		print ('FATS total run time: ',elapsed,'seg for ', features.shape[0],'samples')
		# Writting g5 file with features
		with h5py.File('{}/{}_features.h5'.format(path_to_save, dsname), 'w') as hf:
			hf.create_dataset('features', data=features[..., :-1])
			hf.create_dataset('labels', data=features[..., -1])

def online_features_from_dat(path, path_lcs, path_to_save, name, tokens=[]):
	if tokens == []:
		tokens = np.arange(10, 210, 10)
	
	os.makedirs(path_to_save, exist_ok=True)

	metadata_df = pd.read_csv(path)
	n_classes = len(class_code)

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
				min_v = min_max_by_class['min']
				max_v = min_max_by_class['max']
				normalized_df = (df-min_v)/(max_v-min_v)

				results.append(pool.apply_async(run_fats, args=(normalized_df.values, [class_code[name][row['Class']]])))
				
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
	path_meta = '{}/{}/{}/{}_dataset.dat'.format(main_path, name, name.upper(), name.upper())
	path_lcs  = '{}/{}/{}/LCs/'.format(main_path, name, name.upper())
	path_to_save = '/home/shared/cridonoso/datasets/features/{}'.format(name)
	# path_to_save = '../datasets/features/{}/'.format(name)
	
	rf_features_from_dat(path_meta, path_lcs, path_to_save, name)

	path_test_meta = '/home/shared/cridonoso/datasets/{}/test_curves.csv'.format(name)
	# path_test_meta = '../datasets/features/{}/test_curves.csv'.format(name)

	online_features_from_dat(path_test_meta, path_lcs, path_to_save, name)

	# calculate_features(path, name, n_samples=-1, multiprocessing=True)
	# calculate_online_features(path, name, n_samples=1000, multiprocessing=True)

	# hf = h5py.File('./features/ogle/train/online_features.h5', 'r')
	# print(hf['10'])
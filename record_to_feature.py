from FATSslim import FATS
import os, sys
import multiprocessing as mp
import util_records as data
import numpy as np
import h5py
import tensorflow as tf
import time

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


def run_fats(lc, label):
	fs = FATS.FeatureSpace(Data=['magnitude','time', 'error'], featureList=most_important+harmonics+others)
	results = fs.calculateFeature([lc[:,0], lc[:,1], lc[:,2]])	

	dic = results.result(method='dict')
	array = np.array(list(dic.values()), dtype=np.float32)
	array = np.nan_to_num(np.concatenate([array, label]))

	return array

def calculate_features(path, name, n_samples=-1):
	
	light_curves = data.load_record(path, 1, n_samples=n_samples)

	starttime = time.time()
	processes = []


	start_time = time.time()
	num_cores = mp.cpu_count()
	print ('you are using',num_cores,'cores')
	
	pool = mp.Pool(processes=num_cores)
	
	results = []
	for x, y, m in light_curves:
		lc = tf.boolean_mask(x[0], m[0]).numpy()
		results.append(pool.apply_async(run_fats, args=(lc, y.numpy())))

	features = np.array([p.get() for p in results])
	elapsed = time.time() - start_time
	print ('FATS total run time: ',elapsed,'seg for ', features.shape[0],'samples')

	total_feat = np.array(most_important+harmonics+others)
	with h5py.File('{}.h5'.format(name), 'w') as hf:
		hf.create_dataset('features', data=features)

def calculate_online_features(path, name, tokens=[], n_samples=-1):
	if tokens == []:
		tokens = np.arange(10, 210, 10)
	
	path_to_save = './online_features/{}'.format(name)
	os.makedirs(path_to_save, exist_ok=True)

	light_curves = data.load_record(path, 1, n_samples=n_samples)

	num_cores = mp.cpu_count()
	print('[INFO] Online computation')
	print ('[INFO] Using', num_cores,'cores')


	for lim in tokens:
		starttime = time.time()
		pool = mp.Pool(processes=num_cores)
		results = []
		for i, (x, y, m )in enumerate(light_curves):
			lc = tf.boolean_mask(x[0], m[0]).numpy()[:lim]
			results.append(pool.apply_async(run_fats, args=(lc, y.numpy())))

		features = np.array([p.get() for p in results])
		
		with h5py.File('{}/sub_{}.h5'.format(path_to_save, lim), 'w') as hf:
			hf.create_dataset('features', data=features)

		print('[INFO] {} obs done! {} seconds for {} curves'.format(lim, time.time() - starttime, features.shape[0]))



if __name__ == '__main__':
	# path = '../datasets/records/linear/fold_0/test.tfrecords'
	path = sys.argv[1]
	name = sys.argv[2] # test_0

	# calculate_features(path, name, n_samples=100)
	calculate_online_features(path, name, n_samples=100)

	# hf = h5py.File('test0.h5', 'r')
	# print(hf.keys())
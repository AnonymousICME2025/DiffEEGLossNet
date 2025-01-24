import biosig
import os
import mne
import json
import torch

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


_BCICIV_TYPES = {
	"0x0301": 1,
	"0x0302": 2,
	"0x0303": 3,
	"0x0304": 4
}

# Load the BCICIV2a dataset
def load_bciciv2a(s:int, freq=128):
	datafile = f"{os.path.dirname(__file__)}/BCICIV_2a_gdf/a{s:02d}T.gdf"	
	header = json.loads(biosig.header(datafile))			
	raw = mne.io.read_raw_gdf(datafile)							
	signal = raw.load_data().filter(4,38).resample(freq).get_data()	# Load data, filter, resample, and retrieve data

	labels = [0] * int(header['NumberOfSamples'])		# Initialize the tag list
	for event in header['EVENT']:					
		if event['TYP'] in _BCICIV_TYPES:				
			idx = int(event['POS'] * freq)				# Calculating indexes
			labels[idx] = _BCICIV_TYPES[event['TYP']]
	labels = np.array(labels)						
	results = []		
	for idx in np.argwhere(labels!=0).flatten():		
		x = signal[:,idx+freq//2:idx+freq*4]	# Get the signal
		mu = x.mean(-1,keepdims=True)			# Calculate the mean of a signal
		x = x - mu			
		xmax = x.max(-1,keepdims=True)			
		xmin = x.min(-1,keepdims=True)			
		x = 2 * (x - xmin) / (xmax - xmin) - 1	# Normalize the signal
		y = labels[idx]			
		results.append((x,y,s))			# Add signals, labels and subjects to the result list
	return results	


# Load the BCICIV2b dataset
def load_bciciv2b(s:int, freq=128):
	datafile = f"{os.path.dirname(__file__)}/BCICIV_2b_gdf/a{s:02d}T.gdf"
	header = json.loads(biosig.header(datafile))				
	raw = mne.io.read_raw_gdf(datafile)							
	signal = raw.load_data().filter(4,38).resample(freq).get_data()	

	labels = [0] * int(header['NumberOfSamples'])		
	for event in header['EVENT']:						
		if event['TYP'] in _BCICIV_TYPES:				
			idx = int(event['POS'] * freq)				
			labels[idx] = _BCICIV_TYPES[event['TYP']]	
	labels = np.array(labels)							
	results = []			
	for idx in np.argwhere(labels!=0).flatten():		
		x = signal[:,idx+freq//2:idx+freq*4]	
		mu = x.mean(-1,keepdims=True)			
		x = x - mu			
		xmax = x.max(-1,keepdims=True)			
		xmin = x.min(-1,keepdims=True)			
		x = 2 * (x - xmin) / (xmax - xmin) - 1	
		y = labels[idx]			
		results.append((x,y,s))			
	return results			
		
# File processing

# First, use biosig.header(datafile) to read the header information of the .gdf file and convert it to json format.
# Then, use mne.io.read_raw_gdf(datafile) to read the data of the .gdf file and store it in the raw variable.
# Next, process the data in raw, including loading data, filtering (frequency range is 4-38Hz), resampling (sampling frequency is freq, default is 128Hz), and obtaining the processed data, which is stored in the signal variable.
# Then, initialize the label list labels, and add the corresponding label to labels according to the event type in the header information.
# Next, for the index whose label is not 0, obtain the corresponding signal, calculate the mean of the signal, subtract the mean, and then calculate the maximum and minimum values ​​of the signal, normalize the signal, obtain the label, and add the signal, label, and subject to the result list.
# Define an EEG dataset class, inherited from PyTorch's Dataset class


class EEGDataset(Dataset):
	def __init__(self, n:int, cached=False, ratio=(0.7,0.15,0.15), partition=None):
		self.data = []				
		self.cached = cached		
		self.ratio = ratio			
		self.subject_labels = {(i+1):i for i in range(n)}		
		assert not cached or partition is not None		
		assert partition is None or partition in ['train','val','test']	
		self.partition = partition		

	def __len__(self):
		if self.cached:				
			return len(os.listdir(f"{self.cachedir}/{self.partition}/"))	
		return len(self.data)		

	def __getitem__(self,index):
		if self.cached:				
			return torch.load(f"{self.cachedir}/{self.partition}/sample{index}.pt")	
		return self._parse_data(self.data[index])		

	def _parse_data(self,data):
		x, y, s = data		
		return x, self.class_labels[y], self.subject_labels[s]	

	def cache(self):
		for index, sample in enumerate(self.train):		
			torch.save(sample, f"{self.cachedir}/train/sample{index}.pt")	
		for index, sample in enumerate(self.val):		
			torch.save(sample, f"{self.cachedir}/val/sample{index}.pt")		
		for index, sample in enumerate(self.test):		
			torch.save(sample, f"{self.cachedir}/test/sample{index}.pt")

	def _split(self, ratio):
		signals = [x for x,_,_ in self.data]				
		labels = [(y,s) for _,y,s in self.data]					
		signals_train, signals_valtest, labels_train, labels_valtest = train_test_split(
			signals, labels, train_size=ratio[0], random_state=42, stratify=labels)		
		signals_val, signals_test, labels_val, labels_test = train_test_split(
			signals_valtest, labels_valtest, train_size=(ratio[1]/(ratio[1]+ratio[2])), random_state=42, stratify=labels_valtest)		
		train = [self._parse_data((signals_train[i], labels_train[i][0], labels_train[i][1])) for i in range(len(labels_train))]
		val = [self._parse_data((signals_val[i], labels_val[i][0], labels_val[i][1])) for i in range(len(labels_val))]
		test = [self._parse_data((signals_test[i], labels_test[i][0], labels_test[i][1])) for i in range(len(labels_test))]
		return train, val, test		



class BCICIV2aDataset(EEGDataset):
	def __init__(self, n:int, cached=False, ratio=(0.7,0.15,0.15), partition=None):
		super().__init__(n, cached, ratio, partition)		
		self.class_labels = {1:0, 2:1, 3:2, 4:3}			
		self.cachedir = Path(f"{os.path.dirname(__file__)}/pt_bciciv_2a")	
		if not cached:		
			for i in range(1,n+1):
				self.data += load_bciciv2a(i)		
			self.train, self.val, self.test = self._split(self.ratio)
			self.cache()	


class BCICIV2bDataset(EEGDataset):
	def __init__(self, n:int, cached=False, ratio=(0.7,0.15,0.15), partition=None):
		super().__init__(n, cached, ratio, partition)		
		self.class_labels = {1:0, 2:1}			
		self.cachedir = Path(f"{os.path.dirname(__file__)}/pt_bciciv_2b")	
		if not cached:		
			for i in range(1,n+1):
				self.data += load_bciciv2b(i)		
			self.train, self.val, self.test = self._split(self.ratio)	
			self.cache()	

class GenDataset(Dataset):
	def __init__(self, data, run):
		self.cachedir = Path(f"{os.path.dirname(__file__)}/../sampled/{data}/{run}/")	

	def __len__(self):
		return len(os.listdir(f"{self.cachedir}/")) - 1		

	def __getitem__(self,index):
		x,y,s = torch.load(f"{self.cachedir}/tensor{index}.pt")	
		return x.numpy().squeeze(0), y.item() if y is not None else -1, s.item() if s is not None else -1 

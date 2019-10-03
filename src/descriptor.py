# -- MAIN CLASS TO COMPLETE TASK 1 -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import pickle
import cv2
import os

# -- CLASS TO COMPLETE THE TASK 1 -- #
class GenerateDescriptors():
	"""CLASS::GenerateDescriptors:
		>- Class in charge of computing the descriptors for all the images from a directory."""
	def __init__(self,data_path,img_format='jpg'):
		self.filenames = sorted(glob(datapath+os.sep+'*.'+img_format))
		self.result = {}
	
	def compute_descriptors(self):
		"""METHOD::COMPUTE_DESCRIPTORS:
			Computes for each image on the specified data path the correspondant descriptor."""
		for k,filename in enumerate(self.filenames):
			img = cv2.imread(filename)
			histogram = self._compute_histogram(img)
			feature = self._extract_features(histogram)
			self.result[k] = feature
	
	def save_results(self,out_path,filename):
		"""METHOD::SAVE_RESULTS:
			>- To save the dictionary containing all the descriptors for all the images."""
		with open(out_path+os.sep+filename) as file:
			pickle.dump(self.results,file)
		
	def _compute_histogram(self,img,mask=None,color_space='ycrcb'):
		"""METHOD::COMPUTE_HISTOGRAM:
			>- Returns:  numpy array representing an the histogram of chrominance."""
		if color_space is 'ycrcb':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
		elif color_space is 'lab':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
		else:
			raise NotImplementedError
		return cv2.calcHist([img],[1,2],mask,[256,256],[0,256,0,256])

	def _extract_features(self,histogram,norm='lprob',sub_factor=16):
		"""METHOD::EXTRACT_FEATURES:
			>- Returns: numpy array representing the extracted feature from it's histogram."""
		flat_hist = histogram.flatten()
		if norm is 'lprob':
			norm = np.sum(flat_hist,axis=-1)
		elif norm is 'lmax':
			norm = np.max(flat_hist)
		elif norm is 'l2':
			norm = np.sqrt(np.sum(np.abs(np.power(flat_hist,2)),axis=-1))
		else:
			raise NotImplementedError
		norm_hist = flat_hist/norm
		# Perform average subsampling.
		feature = np.zeros_like(norm_hist[0:-1:sub_factor])
		for i,_ in enumerate(feature[0:-sub_factor]):
			feature[i]=np.mean(norm_hist[i*sub_factor:(i+1)*sub_factor])
		return feature
		 
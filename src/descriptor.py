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
	def __init__(self,data_path,img_format='jpg',masks=False,mask_path=None):
		self.filenames = sorted(glob(data_path+os.sep+'*.'+img_format))
		if masks:
			self.masks = sorted(glob(mask_path+os.sep+'*.png'))
		else:
			self.masks = [None]*len(self.filenames)
		self.result = {}
	
	def compute_descriptors(self):
		"""METHOD::COMPUTE_DESCRIPTORS:
			Computes for each image on the specified data path the correspondant descriptor."""
		print('--- COMPUTING DESCRIPTORS --- ')
		print('-------')
		for k,filename in enumerate(self.filenames):
			img = cv2.imread(filename)
			if self.masks[k] is not None:
				mask = [cv2.imread(self.masks[k],0)]
			else:
				mask = self.masks[k]
			histogram = self._compute_histogram(img,mask)
			feature = self._extract_features(histogram)
			self.result[k] = feature
			print('Image ['+str(k)+'] Computed')
			print('-------')
	
	def save_results(self,out_path,filename):
		"""METHOD::SAVE_RESULTS:
			>- To save the dictionary containing all the descriptors for all the images."""
		with open(out_path+os.sep+filename,'wb') as file:
			pickle.dump(self.results,file)
		print('--- DESCRIPTORS SAVED ---')
		
	def _compute_histogram(self,img,mask,color_space='ycrcb'):
		"""METHOD::COMPUTE_HISTOGRAM:
			>- Returns:  numpy array representing an the histogram of chrominance."""
		if color_space is 'ycrcb':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
		elif color_space is 'lab':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
		else:
			raise NotImplementedError
		return cv2.calcHist([img],[1,2],mask,[256,256],[0,256,0,256])

	def _extract_features(self,histogram,norm='l1',sub_factor=16):
		"""METHOD::EXTRACT_FEATURES:
			>- Returns: numpy array representing the extracted feature from it's histogram."""
		if norm is 'l1':
			norm_hist = cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L1)
		if norm is 'l2':
			norm_hist = cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L2)
		flat_hist = norm_hist.flatten()
		# Perform average subsampling.
		feature = np.zeros_like(flat_hist[0:-1:sub_factor])
		for i,_ in enumerate(feature[0:-sub_factor]):
			feature[i]=np.mean(flat_hist[i*sub_factor:(i+1)*sub_factor])
		return feature
# -- MAIN CLASS TO COMPLETE TASK 1 -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import pickle
import cv2
import os

# -- CLASS TO GENERATE THE DESCRIPTORS FOR EACH IMAGE -- #
class GenerateSubBlockDescriptor():
	"""CLASS::GenerateSubBlockDescriptor:
		>- Class in charge of computing the descriptors for all the images from a directory.
		This class divides the image in sub-blocks to compute the descriptors at each sub-blok and then extend the results"""
	def __init__(self,data_path,img_format='jpg',masks=False,mask_path=None):
		self.filenames = sorted(glob(data_path+os.sep+'*.'+img_format))
		if masks:
			self.masks = sorted(glob(mask_path+os.sep+'*.png'))
		else:
			self.masks = [None]*len(self.filenames)
		self.result = {}

	def compute_descriptors(self,grid_blocks=[3,3],quantify=[12,6,6],color_space='hsv'):
		"""METHOD::COMPUTE_DESCRIPTORS:
			Computes for each image on the specified data path the correspondant descriptor."""
		self.quantify = quantify
		self.color_space = color_space
		print('--- COMPUTING DESCRIPTORS --- ')
		print('-------')
		for k,filename in enumerate(self.filenames):
			img = cv2.imread(filename)
			if self.masks[k] is not None:
				mask = cv2.imread(self.masks[k],0)
			else:
				mask = self.masks[k]
			self.result[k] = self._compute_level(grid_blocks,img,mask)
			print('Image ['+str(k)+'] Computed')
			print('-------')

	def save_results(self,out_path,filename):
		"""METHOD::SAVE_RESULTS:
			>- To save the dictionary containing all the descriptors for all the images."""
		with open(out_path+os.sep+filename,'wb') as file:
			pickle.dump(self.result,file)
		print('--- DESCRIPTORS SAVED ---')
	
	def _compute_level(self,grid_blocks,img,mask):
		"""METHOD::COMPUTE_LEVEL:
			>- Returns the features obtained from each sub division"""
		features = []
		for i in range(grid_blocks[0]):
			for j in range(grid_blocks[1]):
				new_mask = mask
				if mask is not None:
					new_mask = mask[int((i/grid_blocks[0])*mask.shape[0]):int(((i+1)/grid_blocks[0])*mask.shape[0]),
						int((j/grid_blocks[1])*mask.shape[1]):int(((j+1)/grid_blocks[1])*mask.shape[1])]
				new_img = img[int((i/grid_blocks[0])*img.shape[0]):int(((i+1)/grid_blocks[0])*img.shape[0]),
					int((j/grid_blocks[1])*img.shape[1]):int(((j+1)/grid_blocks[1])*img.shape[1])]
				histogram = self._compute_histogram(new_img,new_mask)
				feature = self._extract_features(histogram)
				features.extend(feature)
		return features

	def _compute_histogram(self,img,mask):
		"""METHOD::COMPUTE_HISTOGRAM:
			>- Returns:  numpy array representing an the histogram of chrominance."""
		if self.color_space is 'ycrcb':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
		elif self.color_space is 'lab':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
		elif self.color_space is 'hsv':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		else:
			raise NotImplementedError
		return cv2.calcHist([img],[0,1,2],mask,self.quantify,[0,256,0,256,0,256])

	def _extract_features(self,histogram,norm='l1',sub_factor=1):
		"""METHOD::EXTRACT_FEATURES:
			>- Returns: numpy array representing the extracted feature from its histogram."""
		if norm is 'l1':
			norm_hist = cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L1)
		if norm is 'l2':
			norm_hist = cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L2)
		flat_hist = norm_hist.flatten()
		# Perform average subsampling.
		feature = np.mean(flat_hist.reshape(-1, sub_factor), 1)
		return feature

class GenerateLevelDescriptor(GenerateSubBlockDescriptor):
	"""CLASS::GenerateLevelDescriptor:
		>- Class in charge of computing the descriptors for all the images from a directory."""
	def __init__(self,data_path,img_format='jpg',masks=False,mask_path=None):
		super().__init__(data_path,img_format,masks,mask_path)
	
	def compute_descriptors(self,levels=3,init_quant=[64,32,32],color_space='hsv'):
		self.quantify = np.asarray(init_quant)
		if np.max(self.quantify)
		self.color_space = color_space
		print('--- COMPUTING DESCRIPTORS --- ')
		print('-------')
		for k,filename in enumerate(self.filenames):
			img = cv2.imread(filename)
			if self.masks[k] is not None:
				mask = cv2.imread(self.masks[k],0)
			else:
				mask = self.masks[k]
			features = []
			grid_blocks = np.array([1,1])
			for l in range(level):
				feature = self._compute_level(grid_blocks.tolist(),img,mask)
				features.extend(feature)
				self.quantify/2
				grid_blocks*2
			self.result[k] = features
			print('Image ['+str(k)+'] Computed')
			print('-------')
# -- MAIN CLASS TO COMPLETE TASK 1 -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import pickle
import cv2
import os

# -- CLASS TO GENERATE THE DESCRIPTORS FOR EACH IMAGE -- #
class SubBlockDescriptor():
	"""CLASS::SubBlockDescriptor:
		>- Class in charge of computing the descriptors for all the images from a directory.
		This class divides the image in sub-blocks to compute the descriptors at each sub-blok and then extend the results"""
	def __init__(self,img_list,mask_list,flag=True):
		self.img_list = img_list
		if flag:
			self.mask_list = mask_list
		else:
			self.mask_list = [[None]]*len(self.img_list)
		self.result = {}

	def compute_descriptors(self,grid_blocks=[3,3],quantify=[12,6,6],color_space='hsv'):
		"""METHOD::COMPUTE_DESCRIPTORS:
			Computes for each image on the specified data path the correspondant descriptor."""
		self.quantify = quantify
		self.color_space = color_space
		print('--- COMPUTING DESCRIPTORS --- ')
		for k,images in enumerate(self.img_list):
			self.result[k] = []
			for i,img in enumerate(images):
				self.result[k].append(self._compute_level(grid_blocks,img,self.mask_list[k][i]))
		print('--- DONE --- ')

	def clear_memory(self):
		"""METHOD::CLEAR_MEMORY:
			>- Deletes the memory allocated that stores data to make it more efficient."""
		self.result = {}
	
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
		if self.color_space == 'ycrcb':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
		elif self.color_space == 'lab':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
		elif self.color_space == 'hsv':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		else:
			raise NotImplementedError
		return cv2.calcHist([img],[0,1,2],mask,self.quantify,[0,256,0,256,0,256])

	def _extract_features(self,histogram,norm='l1',sub_factor=1):
		"""METHOD::EXTRACT_FEATURES:
			>- Returns: numpy array representing the extracted feature from its histogram."""
		if norm == 'l1':
			norm_hist = cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L1)
		if norm == 'l2':
			norm_hist = cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L2)
		flat_hist = norm_hist.flatten()
		# Perform average subsampling.
		feature = np.mean(flat_hist.reshape(-1, sub_factor), 1)
		return feature

class LevelDescriptor(SubBlockDescriptor):
	"""CLASS::LevelDescriptor:
		>- Class in charge of computing the descriptors for all the images from a directory."""
	def __init__(self,img_list,mask_list,flag=True):
		super().__init__(img_list,mask_list,flag)
	
	def compute_descriptors(self,levels=3,init_quant=[16,8,8],start=3,jump=2,color_space='hsv'):
		self.quantify = np.asarray(init_quant)
		if np.min(self.quantify)/np.power(2,levels) <= 0:
			raise ValueError('The amount of levels are bigger than the quantification steps.')
		self.color_space = color_space
		print('--- COMPUTING DESCRIPTORS --- ')
		for k,images in enumerate(self.images):
			self.result[k] = []
			for i,img in images:
				features = []
				grid_blocks = np.array([start,start])
				for l in range(levels):
					feature = self._compute_level(grid_blocks.tolist(),img,mask)
					features.extend(feature)
					grid_blocks*jump
				self.result[k].append(features)
		print('--- DONE --- ')
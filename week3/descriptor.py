# -- MAIN CLASS TO COMPLETE TASK 1 -- #

# -- IMPORTS -- #
from skimage import feature as F
from glob import glob
import PIL as pil
import pytesseract
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

	def compute_descriptors(self,grid_blocks=[8,8],quantify=[32,8,8],color_space='hsv'):
		"""METHOD::COMPUTE_DESCRIPTORS:
			Computes for each image on the specified data path the correspondant descriptor."""
		self.quantify = quantify
		self.color_space = color_space
		print('--- COMPUTING DESCRIPTORS --- ')
		for k,images in enumerate(self.img_list):
			self.result[k] = []
			for i,paint in enumerate(images):
				self.result[k].append(self._compute_level(grid_blocks,paint,self.mask_list[k][i]))
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
		hist_range = [0,255,0,255,0,255]
		if self.color_space == 'ycrcb':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
		elif self.color_space == 'lab':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
		elif self.color_space == 'hsv':
			img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			hist_range = [0,179,0,255,0,255]
		else:
			raise NotImplementedError
		return cv2.calcHist([img],[0,1,2],mask,self.quantify,hist_range)

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
	
	def compute_descriptors(self,levels=2,init_quant=[16,8,8],start=3,jump=2,color_space='hsv'):
		self.quantify = np.asarray(init_quant)
		if np.min(self.quantify)/np.power(2,levels) <= 0:
			raise ValueError('The amount of levels are bigger than the quantification steps.')
		self.color_space = color_space
		print('--- COMPUTING DESCRIPTORS --- ')
		for k,images in enumerate(self.images):
			self.result[k] = []
			for i,paint in images:
				features = []
				grid_blocks = np.array([start,start])
				for l in range(levels):
					feature = self._compute_level(grid_blocks.tolist(),paint,mask)
					features.extend(feature)
					grid_blocks*jump
				self.result[k].append(features)
		print('--- DONE --- ')

class TransformDescriptor():
	"""CLASS::TransformDescriptor:
		>- Class in charge of computing the descriptors for all the images from a directory."""
	def __init__(self,img_list,mask_list,flag=True):
		self.img_list = img_list
		if flag:
			self.mask_list = mask_list
		else:
			self.mask_list = [[None]]*len(self.img_list)
		self.result = {}

	def compute_descriptors(self,dct_blocks=8,lbp_blocks=15,transform_type='dct'):
		"""METHOD::COMPUTE_DESCRIPTORS:
			Computes for each image on the specified data path the correspondant descriptor."""
		self.transform_type = transform_type
		self.lbp_blocks = lbp_blocks
		self.dct_blocks = dct_blocks
		print('--- COMPUTING DESCRIPTORS --- ')
		for k,images in enumerate(self.img_list):
			print(str(k)+' out of '+str(len(self.img_list)))
			self.result[k] = []
			for i,paint in enumerate(images):
				self.result[k].append(self._compute_features(paint,self.mask_list[k][i]))
		print('--- DONE --- ')

	def clear_memory(self):
		"""METHOD::CLEAR_MEMORY:
			>- Deletes the memory allocated that stores data to make it more efficient."""
		self.result = {}
	
	def _compute_features(self,img,mask):
		"""METHOD::COMPUTE_FEATURES:
			>- Returns the features obtained."""
		if self.transform_type == 'lbp':
			return self._compute_lbp(img,mask)
		elif self.transform_type == 'dct':
			return self._compute_dct(img,mask)
		elif self.transform_type == 'hog':
			return self._compute_hog(img,mask)
		
	def _compute_lbp(self,img,mask):
		features = []
		img = cv2.resize(img,(500,500))
		if mask is not None:
			mask = cv2.resize(mask,(500,500))
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		for i in range(self.lbp_blocks):
			for j in range(self.lbp_blocks):
				new_mask = mask
				if mask is not None:
					new_mask = mask[int((i/self.lbp_blocks)*mask.shape[0]):int(((i+1)/self.lbp_blocks)*mask.shape[0]),
						int((j/self.lbp_blocks)*mask.shape[1]):int(((j+1)/self.lbp_blocks)*mask.shape[1])]
				new_img = img[int((i/self.lbp_blocks)*img.shape[0]):int(((i+1)/self.lbp_blocks)*img.shape[0]),
					int((j/self.lbp_blocks)*img.shape[1]):int(((j+1)/self.lbp_blocks)*img.shape[1])]
				feature = self._lbp(new_img,new_mask,numPoints=8,radius=2)
				features.extend(feature)
		return features

	def _lbp(self, image, mask, numPoints, radius):
		# lbp = feature.local_binary_pattern(image,numPoints,radius,method="uniform")
		lbp = F.local_binary_pattern(image,numPoints,radius)
		# print(np.max(lbp))
		# lbp = np.uint8(lbp)
		lbp = np.float32(lbp)
		# print(np.max(lbp))
		hist = cv2.calcHist([lbp],[0],mask,[256],[0,255])
		hist = cv2.normalize(hist,hist,norm_type=cv2.NORM_L1)
		hist = hist.flatten()
		return hist
	
	def _compute_dct(self,img,mask,p=0.05):
		features = []
		num_coeff = int(np.power(512/self.dct_blocks,2)*p)
		img = cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		for i in range(self.dct_blocks):
			for j in range(self.dct_blocks):
				new_mask = mask
				if mask is not None:
					new_mask = mask[int((i/self.dct_blocks)*mask.shape[0]):int(((i+1)/self.dct_blocks)*mask.shape[0]),
							int((j/self.dct_blocks)*mask.shape[1]):int(((j+1)/self.dct_blocks)*mask.shape[1])]
				new_img = img[int((i/self.dct_blocks)*img.shape[0]):int(((i+1)/self.dct_blocks)*img.shape[0]),
						int((j/self.dct_blocks)*img.shape[1]):int(((j+1)/self.dct_blocks)*img.shape[1])]
				transform = cv2.dct(np.float32(new_img)/255.0)
				coeff = self._zigzag(transform)[:num_coeff]
				features.extend(coeff)
		return features

	def _zigzag(self,a):
		return np.concatenate([np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-a.shape[0], a.shape[0])])

	def _compute_hog(self,img,mask):
		new_img = cv2.bitwise_and(img,img,mask = mask)
		resized = cv2.resize(new_img,(128,256),cv2.INTER_AREA)
		winSize = (128,256)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 5
		derivAperture = 1
		winSigma = 4.
		histogramNormType = 0
		L2HysThreshold = 2.0000000000000001e-01
		gammaCorrection = 0
		nlevels = 64
		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
		feature = hog.compute(resized,winStride=(8,8),padding=(8,8),locations=None).tolist()
		return [item[0] for item in feature]
		

class TextDescriptor():
	"""CLASS::TextDescriptor:
		>- Class in charge of computing the descriptors for all the images from a directory."""
	def __init__(self,img_list,bbox_list):
		self.img_list = img_list
		self.bbox_list = bbox_list
		
	def compute_descriptors(self):
		print('--- COMPUTING DESCRIPTORS --- ')
		for k,images in enumerate(self.img_list):
			print(str(k)+' out of '+str(len(self.img_list)))
			self.result[k] = []
			for i,paint in enumerate(images):
				self.result[k].append(self._compute_features(paint,self.bbox_list[k][i]))
		print('--- DONE --- ')

	def _compute_features(self,img,bbox):
		features = []
		cropped_text = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
		features.append([pytesseract.image_to_string(pil.Image.fromarray(cropped_text))])
		return features



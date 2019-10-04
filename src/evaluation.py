# -- CLASS TO EVALUATE MASKS BUT ALSO THE CBIR SYSTEM -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import ml_metrics as mlm
import cv2
import os

# -- CLASS TO EVALUATE RESULTS -- #
class EvaluationT1():
	"""CLASS::EvaluationT1:
		>- Class to evaluate method of task 1."""
	def __init__(self,query_res_path,gt_corr_path):
		with open(query_res_path,'rb') as query_res:
			self.query_res = pickle.load(query_res)
		with open(gt_corr_path,'rb') as gt_corrs:
			self.gt_corrs = pickle.load(gt_corrs)
		self.scores = 0
	
	def compute_mapatk(self):
		"""METHOD::COMPUTE_MAPATK:
			>- Computes the MAPatk score for the results obtained."""
		new_gt = []
		for gt in self.gt_corrs:
			new_gt.append(gt[0])
		self.scores = self.MAPatK(new_gt,self.query_res))
		print('The score obtained is: ['+str(self.scores)+'].')
	
	def MAPatK(self,x,y):
		"""
		metrics.mapk.__doc__:

		Computes the mean average precision at k.

		This function computes the mean average prescision at k between two lists
		of lists of items.

		Parameters
		----------
		x : list
			A list of lists of elements that are to be predicted 
			(order doesn't matter in the lists)
		y : list
			A list of lists of predicted elements
			(order matters in the lists)

		Returns
		-------
		score : double
				The mean average precision at k over the input lists
		"""
		return metrics.mapk(x,y)

class EvaluationT5():
	"""CLASS::EvaluationT5:
		>- Class to evaluate the masks of task 5."""
	def __init__(self,res_path,gt_path):
		self.res = sorted(glob(mask_res_path+os.sep+'*.png'))
		self.gt = sorted(glob(gt_path+op.sep+'*.png'))
		self.scores = []

	def compute_fscore(self):
		"""METHOD::COMPUTE_FSCORE:
			>- Uses the function F1_measure to compute the fscore of each pair of masks."""
		for gt,res in zip(self.gt,self.res):
			gt_mask = cv2.imread(gt,0)
			res_mask = cv2.imread(res,0)
			self.scores.append(self.F1_measure(gt_mask,res_mask))
		self.scores = self.scores/len(self.gt)
		print('The score obtained is: ['+str(self.scores)+'].')

	def F1_measure(self,gt,res):
		"""
		This function computes the F1 measure between gt and res.

		Parameters
		----------
		gt : Ground truth binary image as numpy array.

		res : Result binary image as numpy array.

		Returns
		-------
		Integer resulting of computing the F1 measure.
		"""

		TP = np.sum(np.bitwise_and(gt,res) == 1)
		FN = np.sum(np.bitwise_and(gt,(1-res)) == 1)
		FP = np.sum(np.bitwise_and((1-gt),res) == 1)
		TN = np.sum(np.bitwise_and((1-gt),(1-res)) == 1)

		precision = TP/(TP+FP)
		recall = TP/(TP+FN)

		return 2*precision*recall/(precision+recall)

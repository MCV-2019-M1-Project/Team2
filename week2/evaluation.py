# -- CLASS TO EVALUATE MASKS BUT ALSO THE CBIR SYSTEM -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import ml_metrics as metrics
import cv2
import os
import pickle

# -- CLASS TO EVALUATE RESULTS -- #
class EvaluateDescriptors():
	"""CLASS::EvaluateDescriptors:
		>- Class to evaluate method of task 1."""
	def __init__(self,query_results,gt_corr_path):
		self.query_res = query_results
		with open(gt_corr_path,'rb') as gt_corrs:
			self.gt_corrs = pickle.load(gt_corrs)
		self.score = 0
	
	def compute_mapatk(self,limit=1):
		"""METHOD::COMPUTE_MAPATK:
			>- Computes the MAPatk score for the results obtained."""
		query = []
		for img in self.query_res:
			q = []
			for values in img:
				q.append(values[0:limit])
			query.append(q)
		self.score = self.MAPatK(self.gt_corrs,query[0])
	
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

class EvaluateMasks():
	"""CLASS::EvaluateMasks:
		>- Class to evaluate the masks of task 5."""
	def __init__(self,res_path,gt_path):
		self.res = sorted(glob(res_path+os.sep+'*.png'))
		self.gt = sorted(glob(gt_path+os.sep+'*.png'))
		self.score = []
		self.precision = []
		self.recall = []

	def compute_fscore(self):
		"""METHOD::COMPUTE_FSCORE:
			>- Uses the function F1_measure to compute the fscore of each pair of masks."""
		for gt,res in zip(self.gt,self.res):
			gt_mask = cv2.imread(gt,0)
			_, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
			gt_mask = np.asarray(gt_mask,dtype=bool)
			res_mask = cv2.imread(res,0)
			_, res_mask = cv2.threshold(res_mask, 127, 255, cv2.THRESH_BINARY)
			res_mask = np.asarray(res_mask,dtype=bool)
			self.score.append(self.F1_measure(gt_mask,res_mask))
		self.score = np.mean(self.score)
		self.precision = np.mean(self.precision)
		self.recall = np.mean(self.recall)
		print('The F_Score obtained for the masks is: ['+str(self.score)+'].')
		print('The Precision obtained for masks is: ['+str(self.precision)+'].')
		print('The Recall obtained for masks is: ['+str(self.recall)+'].')

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
		self.precision.append(precision)
		self.recall.append(recall)

		return 2*precision*recall/(precision+recall)

class EvaluateIoU():
	"""CLASS::EvaluateIoU:
		>- Class to evaluate the intersection over Union metric for bounding boxes."""
	def __init__(self,bbox_path,gt_path):
		with open(bbox_path,'rb') as file:
			self.bbox = pickle.load(file)
		with open(gt_path,'rb') as file:
			self.gt = pickle.load(file)
		self.score = 0
	
	def compute_iou(self):
	"""METHOD::COMPUTE_IOU:
		>- Uses the function bb_iou to compute the iou of the bounding boxes."""
		iou_result = []
		for gt_box,qr_box in zip(self.gt,self.bbox):
			for gb,qb in zip(gt_box,qr_box):
				iou_result.append(self.bb_iou(gb,qb))
		self.score = np.mean(iou_result)
	
	def bb_iou(boxA, boxB):
	"""METHOD::BB_IOU:
		>- Computes the Interference over Union."""
    	boxA = [int(x) for x in boxA]
    	boxB = [int(x) for x in boxB]

    	xA = max(boxA[0], boxB[0])
    	yA = max(boxA[1], boxB[1])
    	xB = min(boxA[2], boxB[2])
    	yB = min(boxA[3], boxB[3])

    	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    	iou = interArea / float(boxAArea + boxBArea - interArea)

    	return iou
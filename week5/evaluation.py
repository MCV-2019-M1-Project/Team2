# -- CLASS TO EVALUATE MASKS BUT ALSO THE CBIR SYSTEM -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import ml_metrics as metrics
import pickle
import cv2
import os
#from shapely.geometry import Polygon
from random import shuffle

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
		query = [item[0][0:limit] for item in self.query_res]
		self.score = self.MAPatK(self.gt_corrs,query)
	
	def MAPatK(self,x,y):
		"""METHOD::MAPATK:
			>- Calls the function from ML_METRICS"""
		return metrics.mapk(x,y)

class EvaluateMasks():
	"""CLASS::EvaluateMasks:
		>- Class to evaluate the masks of task 5."""
	def __init__(self,masks_res,gt_path):
		self.res = masks_res
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
			# res_mask = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
			res_mask = res
			_, res_mask = cv2.threshold(res_mask, 127, 255, cv2.THRESH_BINARY)
			res_mask = np.asarray(res_mask,dtype=bool)
			self.score.append(self.F1_measure(gt_mask,res_mask))
		self.score = np.mean(self.score)
		self.precision = np.mean(self.precision)
		self.recall = np.mean(self.recall)

	def F1_measure(self,gt,res):
		"""METHOD::F1_MEASURE:
			>- Computes the F1 measure between gt and res"""

		TP = np.sum(np.bitwise_and(gt,res) == 1)
		FN = np.sum(np.bitwise_and(gt,(1-res)) == 1)
		FP = np.sum(np.bitwise_and((1-gt),res) == 1)
		TN = np.sum(np.bitwise_and((1-gt),(1-res)) == 1)

		precision = TP/(TP+FP)
		recall = TP/(TP+FN)
		self.precision.append(precision)
		self.recall.append(recall)

		return 2*precision*recall/(precision+recall)

class EvaluateAngles():
	"""CLASS::EvaluateAngles:
		>- Evaluation of the angle detected."""
	def __init__(self, angle_results, gt_path):
		self.angle_results = angle_results
		with open(gt_path,'rb') as ff:
			self.gt_angle = pickle.load(ff)

	def evaluate(self,degree_margin=1):
		sum_ = 0
		correct = 0
		num_items = 0
		for ind,item in enumerate(self.gt_angle):
			for subitem in item:
				try:
					error = np.abs(((subitem-self.angle_results[ind])+90)%180-90)
				except IndexError:
					pass
				else:
					sum_ += error
					if error < degree_margin:
						correct += 1
					num_items += 1
		self.score = (sum_*1.0/num_items, correct/num_items)
		print('The Mean Angular Error: [{0}]'.format(self.score[0]))
		print('The Precision (degree_margin={0}) ({1}/{2}): [{3}]'.format(degree_margin,correct,num_items,self.score[1]))
		return self.score

class EvaluateIoU():
	"""CLASS::EvaluateIoU:
		>- Class to evaluate the intersection over Union metric for bounding boxes."""
	def __init__(self,bboxes,gt_path):
		self.bbox = bboxes
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
		print('The mean IoU is: [{0}]'.format(self.score))
		return self.score

	def bb_iou(self,boxA, boxB):
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
"""
class EvaluateIoUNonSquare():
	#CLASS::EvaluateIoUNonSquare:
	#	>- Class to evaluate the intersection over Union metric for non-square bounding boxes.
	def __init__(self,bboxes,gt_path):
		self.bbox = bboxes
		with open(gt_path,'rb') as file:
			self.gt = pickle.load(file)
		self.gt = [[[subsubitem[::-1] for subsubitem in subitem[1]][::-1] for subitem in item] for item in self.gt]
		self.score = 0
	
	def compute_iou(self):
		#METHOD::COMPUTE_IOU:
		#	>- Uses the function bb_iou to compute the iou of the bounding boxes.
		max_tries = 20
		self.iou_result = []
		for gt_box,qr_box in zip(self.gt,self.bbox):
			stop = 0
			all_ress = []
			for i in range(max_tries):
				ress = []
				for gb,qb in zip(gt_box,qr_box):
					print(gb,qb,end=" ")
					gb = Polygon(gb)
					qb = Polygon(qb)
					res = gb.intersection(qb).area / gb.union(qb).area
					ress.append(res)
					print(gb.intersection(qb).area / gb.union(qb).area)
				if np.min(ress) > 0.0:
					stop = 1
					break
				else:
					all_ress.append({"ress":ress,"mean":np.mean(ress)})
					shuffle(gt_box)
			if not stop:
				ress = sorted(all_ress,key=lambda x: x["mean"],reverse=True)[0]["ress"]
			for res in ress:
				self.iou_result.append(res)
		self.score = np.mean(self.iou_result)
		print('The mean IoU is: [{0}]'.format(self.score))
		return self.score
		"""

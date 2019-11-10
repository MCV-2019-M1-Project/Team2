# -- CLASS TO EVALUATE MASKS BUT ALSO THE CBIR SYSTEM -- #

# -- IMPORTS -- #
from glob import glob
import numpy as np
import ml_metrics as metrics
import pickle
import cv2
import os

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
				error = np.abs(((subitem-self.angle_results[ind])+90)%180-90)
				sum_ += error
				if error < degree_margin:
					correct += 1
				num_items += 1
		self.score = (sum_*1.0/num_items, correct/num_items)
		print('The Mean Angular Error: [{0}]'.format(self.score[0]))
		print('The Precision ({0}/{1}): [{2}]'.format(correct,num_items,self.score[1]))
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
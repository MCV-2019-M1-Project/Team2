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
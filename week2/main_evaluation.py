# -- IMPORTS -- #
from evaluation import EvaluateMasks, EvaluateDescriptors, EvaluateIoU
import numpy as np
import pickle
import cv2
import os

# -- DIRECTORIES -- #
res_root = '../results'
qs1_mask = '../results/QS1W2'
qs2_mask = '../results/QS2W2'
qs1_w2 = '../qsd1_w2'
qs2_w2 = '../qsd2_w2'

def evaluate():
	# -- OPEN FILES -- #
	print('Opening files')
	with open(res_root+os.sep+'qs1_bbox.pkl','rb') as f:
		qs1_bbox = pickle.load(f)
	with open(res_root+os.sep+'qs2_bbox.pkl','rb') as f:
		qs2_bbox = pickle.load(f)
	with open(res_root+os.sep+'qs1_result.pkl','rb') as f:
		qs1_result = pickle.load(f)
	with open(res_root+os.sep+'qs2_result.pkl','rb') as f:
		qs2_result = pickle.load(f)
	print('Done')

	# -- EVALUATE BBOX QS1 -- #
	print('Evaluating QS1:')
	qs1_box_eval = EvaluateIoU(qs1_bbox,qs1_w2+os.sep+'text_boxes.pkl')
	qs1_mask_eval = EvaluateMasks(qs1_mask,qs1_w2)
	qs1_desc_eval = EvaluateDescriptors(qs1_result,qs1_w2+os.sep+'gt_corresps.pkl')
	qs1_box_eval.compute_iou()
	qs1_mask_eval.compute_fscore()
	qs1_desc_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs1_desc_eval.score)+']')
	qs1_desc_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs1_desc_eval.score)+']')
	print('BBOX IOU: ['+str(qs1_box_eval.score)+']')
	print('MASK FSCORE: ['+str(qs1_mask_eval.score)+']')
	print('Done')

	# -- EVALUATE BBOX QS2 -- #
	print('Evaluating QS2:')
	qs2_box_eval = EvaluateIoU(qs2_bbox,qs2_w2+os.sep+'text_boxes.pkl')
	qs2_mask_eval = EvaluateMasks(qs2_mask,qs2_w2)
	qs2_desc_eval = EvaluateDescriptors(qs2_result,qs2_w2+os.sep+'gt_corresps.pkl')
	qs2_box_eval.compute_iou()
	qs2_mask_eval.compute_fscore()
	qs2_desc_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs2_desc_eval.score)+']')
	qs2_desc_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs2_desc_eval.score)+']')
	print('BBOX IOU: ['+str(qs2_box_eval.score)+']')
	print('MASK FSCORE: ['+str(qs2_mask_eval.score)+']')
	print('Done')
	
if __name__ == '__main__':
	evaluate()
from glob import glob
from descriptor import SubBlockDescriptor,TransformDescriptor
from descriptor import TextDescriptor, CombinedDescriptor
from searcher import Searcher,SearcherText,SearcherCombined
from evaluation import EvaluateDescriptors
import numpy as np
import pickle
import cv2
import os

# -- DIRECTORIES -- #
res_root = "../results"
db = "../bbdd"
dbt = "../bbdd_text"
qs1_w3 = "../qsd1_w3"
qs2_w3 = "../qsd2_w3"

def get_text():
	text = sorted(glob(dbt+os.sep+'*.txt'))
	db_text = {}
	for k,path in enumerate(text):
		db_text[k] = []
		with open(path,'r') as ff:
			line = ff.readline()
			if not line:
				db_text[k].append([line])
			else:
				line = line.split(',')
				db_text[k].append([line[0][2:-1]])
	return db_text

def test_qs1():
	print('GET DB')
	db_text = get_text()
	db_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(db,"*.jpg")))]
	print('done')
	
	# -- READ -- #
	print('READ FILES')
	with open(res_root+os.sep+'denoised.pkl','rb') as ff:
		qs1_images = pickle.load(ff)
	with open(res_root+os.sep+'qs1_bbox.pkl','rb') as ff:
		qs1_bbox = pickle.load(ff)
	with open(res_root+os.sep+'qs1_mask.pkl','rb') as ff:
		qs1_mask = pickle.load(ff)
	print('done')

	# -- TEXT -- #
	print('computing text descriptors')
	qs_desc = TextDescriptor(qs1_images,qs1_bbox)
	qs_desc.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = SearcherText(db_text,qs_desc.result)
	qs_desc.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs1_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- COLOR -- #
	print('computing color descriptors')
	db_desc_col = SubBlockDescriptor(db_images,None)
	db_desc_col.compute_descriptors()
	qs_desc_col = SubBlockDescriptor(qs1_images,qs1_mask)
	qs_desc_col.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_col.result,qs_desc_col.result)
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs1_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- TRANSFORM -- #
	print('computing combined descriptors')
	db_desc_trans = TransformDescriptor(db_images,None,None)
	db_desc_trans.compute_descriptors()
	qs_desc_trans = TransformDescriptor(qs1_images,qs1_mask,None)
	qs_desc_trans.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_trans.result,qs_desc_trans.result)
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs1_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- SEARCH -- #
	qs_searcher = SearcherCombined(db_desc_col.result,qs_desc_col.result,db_desc_trans.result,qs_desc_trans.result)
	db_desc_col.clear_memory()
	qs_desc_col.clear_memory()
	db_desc_trans.clear_memory()
	qs_desc_trans.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs1_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

def test_qs2():
	print('GET DB')
	db_text = get_text()
	db_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(db,"*.jpg")))]
	print('done')
	
	# -- READ -- #
	print('READ FILES')
	with open(res_root+os.sep+'qs2denoised.pkl','rb') as ff:
		qs2_images = pickle.load(ff)
	with open(res_root+os.sep+'qs2_bbox.pkl','rb') as ff:
		qs2_bbox = pickle.load(ff)
	with open(res_root+os.sep+'qs2_mask.pkl','rb') as ff:
		qs2_mask = pickle.load(ff)
	print('done')

	# -- TEXT -- #
	print('computing text descriptors')
	qs_desc = TextDescriptor(qs2_images,qs2_bbox)
	qs_desc.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = SearcherText(db_text,qs_desc.result)
	qs_desc.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs2_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- COLOR -- #
	print('computing color descriptors')
	db_desc_col = SubBlockDescriptor(db_images,None)
	db_desc_col.compute_descriptors()
	qs_desc_col = SubBlockDescriptor(qs2_images,qs2_mask)
	qs_desc_col.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_col.result,qs_desc_col.result)
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs2_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- TRANSFORM -- #
	print('computing combined descriptors')
	db_desc_trans = TransformDescriptor(db_images,None,None)
	db_desc_trans.compute_descriptors()
	qs_desc_trans = TransformDescriptor(qs2_images,qs2_mask,None)
	qs_desc_trans.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc_trans.result,qs_desc_trans.result)
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs2_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- SEARCH -- #
	qs_searcher = SearcherCombined(db_desc_col.result,qs_desc_col.result,db_desc_trans.result,qs_desc_trans.result)
	db_desc_col.clear_memory()
	qs_desc_col.clear_memory()
	db_desc_trans.clear_memory()
	qs_desc_trans.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs2_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

if __name__ == "__main__":
	test_qs1()
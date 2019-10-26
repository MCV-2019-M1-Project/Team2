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
	db_desc = SubBlockDescriptor(db_images,None)
	db_desc.compute_descriptors()
	qs_desc = SubBlockDescriptor(qs2_images,qs2_mask)
	qs_desc.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc.result,qs_desc.result)
	db_desc.clear_memory()
	qs_desc.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs2_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- COMBINED-- #
	print('computing combined descriptors')
	db_desc = CombinedDescriptor(db_images,None,None)
	db_desc.compute_descriptors()
	qs_desc = CombinedDescriptor(qs2_images,qs2_mask,qs2_bbox)
	qs_desc.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = Searcher(db_desc.result,qs_desc.result)
	db_desc.clear_memory()
	qs_desc.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs2_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

	# -- COMBINED-- #
	print('computing combined descriptors')
	db_desc1 = SubBlockDescriptor(db_images,None)
	db_desc1.compute_descriptors()
	qs_desc1 = SubBlockDescriptor(qs1_images,qs1_mask)
	qs_desc1.compute_descriptors()
	db_desc2 = TransformDescriptor(db_images,None,None)
	db_desc2.compute_descriptors()
	qs_desc2 = TransformDescriptor(qs1_images,qs1_mask,None)
	qs_desc2.compute_descriptors()
	# -- SEARCH -- #
	qs_searcher = SearcherCombined(db_desc1.result,qs_desc1.result,db_desc2.result,qs_desc2.result)
	db_desc1.clear_memory()
	qs_desc1.clear_memory()
	db_desc2.clear_memory()
	qs_desc2.clear_memory()
	qs_searcher.search(limit=3)
	print("Done.")
	qs_eval = EvaluateDescriptors(qs_searcher.result,os.path.join(qs1_w3,'gt_corresps.pkl'))
	qs_eval.compute_mapatk(limit=1)
	print('DESC MAP1: ['+str(qs_eval.score)+']')
	qs_eval.compute_mapatk(limit=3)
	print('DESC MAP3: ['+str(qs_eval.score)+']')
	print('done')

if __name__ == "__main__":
	test_qs1()
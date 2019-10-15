# -- IMPORTS -- #

from descriptor import SubBlockDescriptor, LevelDescriptor
from searcher import Searcher
from evaluation import EvaluationT1
import argparse
import os

# -- EXAMPLE OF EXECUTION -- #

""" TESTING METHOD 1: 
	python3 test_descriptors.py -o 0 -s int int{[2,2]} -q int int int{[32,32,32]} -c string_color_space
	TESTING METHOD 2:
	python3 test_descriptors.py -o 1 -l int{number of levels} -q int int int{[32,32,32]} -c string_color_space"""

# -- DIRECTORIES -- #
db = '../database'
qs1_w1 = '../qsd1_w1'
qs2_w1 = '../qsd2_w1'
qs1_w2 = '../qsd1_w2'
qs2_w2 = '../qsd2_w2'
mask_root = '../results/QS2_masks'
res_root = '../results'

# -- PARAMETERS  #
def get_arguments():
	parser = argparse.ArgumentParser(description='descriptor')
	parser.add_argument('-o','--option',type=int,required=True,choices=[0,1])
	parser.add_argument('-s','--start',type=int,nargs='+')
	parser.add_argument('-l','--level',type=int)
	parser.add_argument('-j','--jump',type=int)
	parser.add_argument('-q','--quantize',type=int,nargs='+',required=True)
	parser.add_argument('-c','--color',type=str,required=True)
	return parser.parse_args()

def test_sub_blocks():
	start = [2,3,4,5,8,10]
	quant = [[12,6,6],[16,8,8],[24,12,12]]
	color = ['hsv']
	res_q1 = []
	res_q2 = []
	for s in start:
		for q in quant:
			print('--- # -- ')
			db_desc = SubBlockDescriptor(db)
			q1_desc = SubBlockDescriptor(qs1_w1)
			q2_desc = SubBlockDescriptor(qs2_w1,masks=True,mask_path=mask_root)
			db_desc.compute_descriptors(grid_blocks=[s,s],quantify=q,color_space=color[0])
			q1_desc.compute_descriptors(grid_blocks=[s,s],quantify=q,color_space=color[0])
			q2_desc.compute_descriptors(grid_blocks=[s,s],quantify=q,color_space=color[0])
			db_desc.save_results(res_root,'db_sub.pkl')
			q1_desc.save_results(res_root,'q1_sub.pkl')
			q2_desc.save_results(res_root,'q2_sub.pkl')
			# -- SEARCH -- #
			q1_search = Searcher(res_root+os.sep+'db_sub.pkl',res_root+os.sep+'q1_sub.pkl')
			q2_search = Searcher(res_root+os.sep+'db_sub.pkl',res_root+os.sep+'q2_sub.pkl')
			q1_search.search(limit=3)
			q2_search.search(limit=3)
			q1.save_results(res_root,'q1_sres.pkl')
			q2.save_results(res_root,'q2_sres.pkl')
			# -- EVALUATION -- #
			q1_eval = EvaluationT1(res_root+os.sep+'q1_sres.pkl',qs1_w1+os.sep+'gt.corresps.pkl')
			q2_eval = EvaluationT1(res_root+os.sep+'q2_sres.pkl',qs2_w1+os.sep+'gt.corresps.pkl')
			res_q1.append({'s':s,'q':q,'c':color[0],'r':q1_eval.compute_mapatk(limit=1)})
			res_q2.append({'s':s,'q':q,'c':color[0],'r':q2_eval.compute_mapatk(limit=1)})
			print('--- # -- ')
	with open('final_sub_res.pkl','wb') as f:
		pickle.dump(res_q1,f)
		pickle.dump(res_q2,f)

def test_level_desc():
	level = [2,3]
	start = [2,3,5,8,10]
	jump = [2,4]
	quant = [[12,6,6],[16,8,8],[24,12,12]]
	color = ['hsv']
	res_q1 = []
	res_q2 = []
	for l in level:
		for s in start:
			for j in jump:
				for q in quant:
					print('--- # -- ')
					db_desc = LevelDescriptor(db)
					q1_desc = LevelDescriptor(qs1_w1)
					q2_desc = LevelDescriptor(qs2_w1,masks=True,mask_path=mask_root)
					db_desc.compute_descriptors(levels=l,init_quant=q,start=s,jump=j,color_space=color[0])
					q1_desc.compute_descriptors(levels=l,init_quant=q,start=s,jump=j,color_space=color[0])
					q2_desc.compute_descriptors(levels=l,init_quant=q,start=s,jump=j,color_space=color[0])
					db_desc.save_results(res_root,'db_lev.pkl')
					q1_desc.save_results(res_root,'q1_lev.pkl')
					q2_desc.save_results(res_root,'q2_lev.pkl')
					# -- SEARCH -- #
					q1_search = Searcher(res_root+os.sep+'db_lev.pkl',res_root+os.sep+'q1_lev.pkl')
					q2_search = Searcher(res_root+os.sep+'db_lev.pkl',res_root+os.sep+'q2_lev.pkl')
					q1_search.search(limit=3)
					q2_search.search(limit=3)
					q1.save_results(res_root,'q1_lres.pkl')
					q2.save_results(res_root,'q2_lres.pkl')
					# -- EVALUATION -- #
					q1_eval = EvaluationT1(res_root+os.sep+'q1_lres.pkl',qs1_w1+os.sep+'gt.corresps.pkl')
					q2_eval = EvaluationT1(res_root+os.sep+'q2_lres.pkl',qs2_w1+os.sep+'gt.corresps.pkl')
					res_q1.append({'s':s,'q':q,'c':color[0],'j':j,'l':l,'r':q1_eval.compute_mapatk(limit=1)})
					res_q2.append({'s':s,'q':q,'c':color[0],'j':j,'l':l,'r':q2_eval.compute_mapatk(limit=1)})
					print('--- # -- ')
	with open('final_lev_res.pkl','wb') as f:
		pickle.dump(res_q1,f)
		pickle.dump(res_q2,f)
		
if __name__ == '__main__':
	print('--- SUB-BLOCKS -- ')
	test_sub_blocks()
	print('--- LEVEL -- ')
	test_level_desc()
	
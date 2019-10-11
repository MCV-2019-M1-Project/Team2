# -- IMPORTS -- #

from descriptor import SubBlockDescriptor, LevelDescriptor
from searcher import Searcher
from evaluation import EvaluationT1,EvaluationT5
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
    parser.add_argument('-o','--option',type=int,required=True)
    parser.add_argument('-s','--size',type=int,nargs='+')
    parser.add_argument('-l','--level',type=int)
    parser.add_argument('-q','--quantize',type=int,nargs='+',required=True)
    parser.add_argument('-c','--color',type=str,required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    if args.option == 0:
        if args.size == None:
            raise ValueError('The input -s needs a value')
        # -- GENERATE DESCRIPTORS FOR DB -- #
        db_descriptor = SubBlockDescriptor(db)
        db_descriptor.compute_descriptors(grid_blocks=args.size,quantify=args.quantize,color_space=args.color)
        db_descriptor.save_results(res_root,'db_sub.pkl')

        # -- GENERATE DECRIPTORS FOR QS1 -- #
        qs1_descriptor = SubBlockDescriptor(qs1_w1)
        qs1_descriptor.compute_descriptors(grid_blocks=args.size,quantify=args.quantize,color_space=args.color)
        qs1_descriptor.save_results(res_root,'qs1_sub.pkl')

        # -- SEARCH MOST SIMILAR FOR QS1 -- #
        qs1_searcher = Searcher(data_path=res_root+os.sep+"db_sub.pkl",query_path=res_root+os.sep+"qs1_sub.pkl")
        qs1_searcher.search(limit=3)
        qs1_searcher.save_results(res_root,"qs1_rsub.pkl")

        # -- EVALUATE RESULTS FOR TASK1 FOR QS1 -- #
        eval_t1 = EvaluationT1(query_res_path=res_root+os.sep+"qs1_rsub.pkl",gt_corr_path=res_root+os.sep+"gt_corresps1.pkl")

        # -- GENERATE DECRIPTORS FOR QS2 -- #
        qs2_descriptor = SubBlockDescriptor(qs2_w1,masks=True,mask_path=mask_root)
        qs2_descriptor.compute_descriptors(grid_blocks=args.size,quantify=args.quantize,color_space=args.color)
        qs2_descriptor.save_results(res_root,'qs2_sub.pkl')

        # -- SEARCH MOST SIMILAR FOR QS2 -- #
        qs2_searcher = Searcher(data_path=res_root+os.sep+"db_sub.pkl",query_path=res_root+os.sep+"qs2_sub.pkl")
        qs2_searcher.search(limit=3)
        qs2_searcher.save_results(res_root,"qs2_rsub.pkl")

        # -- EVALUATE RESULTS FOR TASK1 FOR QS2 -- #
        eval_t2 = EvaluationT1(query_res_path=res_root+os.sep+"qs2_rsub.pkl",gt_corr_path=res_root+os.sep+"gt_corresps2.pkl")

        # -- SHOW RESULTS -- #
        eval_t1.compute_mapatk(limit=1)
        eval_t1.compute_mapatk(limit=3)
        eval_t2.compute_mapatk(limit=1)
        eval_t2.compute_mapatk(limit=3)
    else:
        if args.level == None:
            raise ValueError('The input -l needs a value')
        # -- GENERATE DESCRIPTORS FOR DB -- #
        db_descriptor = LevelDescriptor(db)
        db_descriptor.compute_descriptors(levels=args.level,init_quant=args.quantize,color_space=args.color)
        db_descriptor.save_results(res_root,'db_lev.pkl')

        # -- GENERATE DECRIPTORS FOR QS1 -- #
        qs1_descriptor = LevelDescriptor(qs1_w1)
        qs1_descriptor.compute_descriptors(levels=args.level,init_quant=args.quantize,color_space=args.color)
        qs1_descriptor.save_results(res_root,'qs1_lev.pkl')

        # -- SEARCH MOST SIMILAR FOR QS1 -- #
        qs1_searcher = Searcher(data_path=res_root+os.sep+"db_lev.pkl",query_path=res_root+os.sep+"qs1_lev.pkl")
        qs1_searcher.search(limit=3)
        qs1_searcher.save_results(res_root,"qs1_rlev.pkl")

        # -- EVALUATE RESULTS FOR TASK1 FOR QS1 -- #
        eval_t1 = EvaluationT1(query_res_path=res_root+os.sep+"qs1_rlev.pkl",gt_corr_path=res_root+os.sep+"gt_corresps1.pkl")

        # -- GENERATE DECRIPTORS FOR QS2 -- #
        qs2_descriptor = LevelDescriptor(qs2_w1,masks=True,mask_path=mask_root)
        qs2_descriptor.compute_descriptors(levels=args.level,init_quant=args.quantize,color_space=args.color)
        qs2_descriptor.save_results(res_root,'qs2_lev.pkl')

        # -- SEARCH MOST SIMILAR FOR QS2 -- #
        qs2_searcher = Searcher(data_path=res_root+os.sep+"db_lev.pkl",query_path=res_root+os.sep+"qs2_lev.pkl")
        qs2_searcher.search(limit=3)
        qs2_searcher.save_results(res_root,"qs2_rlev.pkl")

        # -- EVALUATE RESULTS FOR TASK1 FOR QS2 -- #
        eval_t2 = EvaluationT1(query_res_path=res_root+os.sep+"qs2_rlev.pkl",gt_corr_path=res_root+os.sep+"gt_corresps2.pkl")

        # -- SHOW RESULTS -- #
        eval_t1.compute_mapatk(limit=1)
        eval_t1.compute_mapatk(limit=3)
        eval_t2.compute_mapatk(limit=1)
        eval_t2.compute_mapatk(limit=3)
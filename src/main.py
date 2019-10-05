# -- MAIN SCRIPT -- #

# -- IMPORTS -- #
from descriptor import GenerateDescriptors, GenerateDescriptorsGrid
from searcher import Searcher
from evaluation import EvaluationT1,EvaluationT5
import os

# -- CONSTANTS -- #
qs1_root = "../qsd1"
qs2_root = "../qsd2"
masks_path = "../qsd2_masks2"
db_root = "../bbdd"
results_root = "../results"

# -- PROGRAM -- #
if __name__ == '__main__':
    # -- GENERATE DESCRIPTORS FOR DB -- #
    db_descriptor = GenerateDescriptorsGrid(db_root)
    db_descriptor.compute_descriptors()
    db_descriptor.save_results(results_root,'db_desc.pkl')

    # -- GENERATE DECRIPTORS FOR QS1 -- #
    qs1_descriptor = GenerateDescriptorsGrid(qs1_root)
    qs1_descriptor.compute_descriptors()
    qs1_descriptor.save_results(results_root,'qs1_desc.pkl')

    # -- GENERATE DESCRIPTORS FOR QS2 -- #
    # -- OBTAIN MASK -- #
    qs2_descriptor = GenerateDescriptorsGrid(qs2_root,masks=True,mask_path=masks_path)
    qs2_descriptor.compute_descriptors()
    qs2_descriptor.save_results(results_root,'qs2_desc.pkl')

    # -- SEARCH MOST SIMILAR FOR QS1 -- #
    searcher = Searcher(data_path=results_root+os.sep+"db_desc.pkl",
                        query_path=results_root+os.sep+"qs1_desc.pkl")
    searcher.search(limit=10)
    searcher.save_results(results_root,"qs1_results.pkl")

    # -- SEARCH MOST SIMILAR FOR QS2 -- #
    searcher = Searcher(data_path=results_root+os.sep+"db_desc.pkl",
                        query_path=results_root+os.sep+"qs2_desc.pkl")
    searcher.search(limit=10)
    searcher.save_results(results_root,"qs2_results.pkl")

    # -- EVALUATE RESULTS FOR TASK1 FOR QS1 -- #
    eval_t1 = EvaluationT1(query_res_path=results_root+os.sep+"qs1_results.pkl",
                           gt_corr_path=results_root+os.sep+"gt_corresps1.pkl")
    eval_t1.compute_mapatk()

    # -- EVALUATE RESULTS FOR TASK1 FOR QS2 -- #
    eval_t1 = EvaluationT1(query_res_path=results_root+os.sep+"qs2_results.pkl",
                           gt_corr_path=results_root+os.sep+"gt_corresps2.pkl")
    eval_t1.compute_mapatk()

    # -- EVALUATE F1-SCORE FOR THE QS2 MASKS -- #
    eval_t5 = EvaluationT5(res_path=masks_path,
                           gt_path=qs2_root)
    eval_t5.compute_fscore()
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
available = [0,1,2]

# -- PROGRAM -- #
if __name__ == '__main__':

    option = int(input('Run Mode:\n\t (0):Descriptors & Retrieval.\n\t (1):Evaluation.\n\t (2):All.'))
    if option not in available:
        raise Exception('This option is not supported.')

    # -- COMPUTE DESCRIPTORS AND RETRIEVE -- #
    if option is (0 or 2):
        # -- GENERATE DESCRIPTORS FOR DB -- #
        db_descriptor = GenerateDescriptorsGrid(db_root)
        db_descriptor.compute_descriptors()
        db_descriptor.save_results(results_root,'db_desc.pkl')

        # -- GENERATE DECRIPTORS FOR QS1 -- #
        qs1_descriptor = GenerateDescriptorsGrid(qs1_root)
        qs1_descriptor.compute_descriptors()
        qs1_descriptor.save_results(results_root,'qs1_desc.pkl')

        # -- OBTAIN MASK -- #
        # -- GENERATE DESCRIPTORS FOR QS2 -- #
        qs2_descriptor = GenerateDescriptorsGrid(qs2_root,masks=True,mask_path=masks_path)
        qs2_descriptor.compute_descriptors()
        qs2_descriptor.save_results(results_root,'qs2_desc.pkl')

        # -- SEARCH MOST SIMILAR FOR QS1 -- #
        qs1_searcher = Searcher(data_path=results_root+os.sep+"db_desc.pkl",
                            query_path=results_root+os.sep+"qs1_desc.pkl")
        qs1_searcher.search(limit=10)
        qs1_searcher.save_results(results_root,"qs1_results.pkl")

        # -- SEARCH MOST SIMILAR FOR QS2 -- #
        qs2_searcher = Searcher(data_path=results_root+os.sep+"db_desc.pkl",
                            query_path=results_root+os.sep+"qs2_desc.pkl")
        qs2_searcher.search(limit=10)
        qs2_searcher.save_results(results_root,"qs2_results.pkl")

    # -- EVALUATE SYSTEMS -- #
    if option is (1 or 2):
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
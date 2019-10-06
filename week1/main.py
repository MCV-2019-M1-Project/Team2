# -- MAIN SCRIPT -- #

# -- IMPORTS -- #
from descriptor import GenerateDescriptors, GenerateDescriptorsGrid
from searcher import Searcher
from evaluation import EvaluationT1,EvaluationT5
import os

# -- CONSTANTS -- #
qs1_root = "../qs1t"
qs2_root = "../qs2"
masks_path = "../results/QS2_masks"
db_root = "../database"
results_root = "../results"
available = [0,1,2]

# -- PROGRAM -- #
if __name__ == '__main__':

    option = int(input('Specify the dataset:\n\t Both(0):\n\t QS1(1):\n\t QS2(2):\n\t Input:'))
    if option not in available:
        # -- EVALUATE F1-SCORE FOR THE QS2 MASKS -- #
        eval_t5 = EvaluationT5(res_path=masks_path,
                            gt_path=qs2_root)
        eval_t5.compute_fscore()
        raise Exception('Option not supported')

    db_descriptor = GenerateDescriptorsGrid(db_root)
    # -- TEST QS1 -- #
    if option is 0 or option is 1:
        # -- GENERATE DESCRIPTORS FOR DB -- #
        db_descriptor.compute_descriptors(grid_blocks=[10,10],quantify=[16,8,8])
        db_descriptor.save_results(results_root,'db_desc1.pkl')

        # -- GENERATE DECRIPTORS FOR QS1 -- #
        qs1_descriptor = GenerateDescriptorsGrid(qs1_root)
        qs1_descriptor.compute_descriptors(grid_blocks=[10,10],quantify=[16,8,8])
        qs1_descriptor.save_results(results_root,'qs1_desc.pkl')

        # -- SEARCH MOST SIMILAR FOR QS1 -- #
        qs1_searcher = Searcher(data_path=results_root+os.sep+"db_desc1.pkl",
                            query_path=results_root+os.sep+"qs1_desc.pkl")
        qs1_searcher.search(limit=10)
        qs1_searcher.save_results(results_root,"result.pkl")

        # -- EVALUATE RESULTS FOR TASK1 FOR QS1 -- #
        eval_t1 = EvaluationT1(query_res_path=results_root+os.sep+"qs1_results.pkl",
                            gt_corr_path=results_root+os.sep+"gt_corresps1.pkl")
        eval_t1.compute_mapatk()
    if option is 0 or option is 2:
        # -- GENERATE DESCRIPTORS FOR DB -- #
        db_descriptor.compute_descriptors(grid_blocks=[10,10],quantify=[16,8,8])
        db_descriptor.save_results(results_root,'db_desc2.pkl')

        # -- OBTAIN MASK -- #
        # -- GENERATE DESCRIPTORS FOR QS2 -- #
        qs2_descriptor = GenerateDescriptorsGrid(qs2_root,masks=True,mask_path=masks_path)
        qs2_descriptor.compute_descriptors(grid_blocks=[10,10],quantify=[16,8,8])
        qs2_descriptor.save_results(results_root,'qs2_desc.pkl')

        # -- SEARCH MOST SIMILAR FOR QS2 -- #
        qs2_searcher = Searcher(data_path=results_root+os.sep+"db_desc2.pkl",
                            query_path=results_root+os.sep+"qs2_desc.pkl")
        qs2_searcher.search(limit=10)
        qs2_searcher.save_results(results_root,"result.pkl")

        # -- EVALUATE RESULTS FOR TASK1 FOR QS2 -- #
        eval_t1 = EvaluationT1(query_res_path=results_root+os.sep+"qs2_results.pkl",
                            gt_corr_path=results_root+os.sep+"gt_corresps2.pkl")
        eval_t1.compute_mapatk()

        # -- EVALUATE F1-SCORE FOR THE QS2 MASKS -- #
        eval_t5 = EvaluationT5(res_path=masks_path,
                            gt_path=qs2_root)
        eval_t5.compute_fscore()
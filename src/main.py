# -- MAIN SCRIPT -- #

# -- IMPORTS -- #
from descriptor import GenerateDescriptors
#from searcher import Searcher
#from evaluation import EvaluationT1,EvaluationT5
import os

# -- CONSTANTS -- #
qs1_root = '../qs1'
qs2_root = '../qs2'
db_root = '../database'
results_root = '../results'

# -- PROGRAM -- #
if __name__ == '__main__':
    # -- GENERATE DESCRIPTORS FOR DB -- #
    db_descriptor = GenerateDescriptors(db_root)
    db_descriptor.compute_descriptors()
    db_descriptor.save_results(results_root,'db_desc.pkl')
    # -- GENERATE DECRIPTORS FOR QS1 -- #
    qs1_descriptor = GenerateDescriptors(qs1_root)
    qs1_descriptor.compute_descriptors()
    qs1_descriptor.save_results(results_root,'qs1_desc.pkl')
    # -- GENERATE DESCRIPTORS FOR QS2 -- #
    # -- OBTAIN MASK -- #
    #qs2_descriptor = GenerateDescriptors(qs2_root,masks=True,mask_path='../results/masks1')
    #qs2_descriptor.compute_descriptors()
    #qs2_descriptor.save_results(results_root,'qs2_desc.pkl')

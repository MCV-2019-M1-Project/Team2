# -- MAIN SCRIPT -- #

# -- IMPORTS -- #
from .descriptor import GenerateDescriptors
from .searcher import Searcher
from .evaluation import EvaluationT1,EvaluationT5
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
    db_descriptor.save_results(results_root,'db_descriptor.pkl')
    # -- GENERATE DECRIPTORS FOR QS1 -- #
    #qs1_descriptor = GenerateDescriptors(qs1_root)
    # -- REMOVE BACKGROUND -- #
    # -- GENERATE DESCRIPTORS FOR QS2 -- #

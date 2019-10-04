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


# -- PROGRAM -- #
if __name__ == '__main__':
    # -- GENERATE DESCRIPTORS FOR DB -- #
    db_descriptor = GenerateDescriptors(db_root)
    # -- GENERATE DECRIPTORS FOR QS1 -- #
    #qs1_descriptor = GenerateDescriptors(qs1_root)
    # -- REMOVE BACKGROUND -- #
    # -- GENERATE DESCRIPTORS FOR QS2 -- #

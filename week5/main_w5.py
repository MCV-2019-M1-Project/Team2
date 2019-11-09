# -- SCRIPT TO FIND THE BEST DESCRIPTORS AND TEST VARIOUS THINGS -- #

# -- IMPORTS -- #
from noise import Denoiser
from detect_orientation import Orientation
from evaluation import EvaluateAngles
from glob import glob
import numpy as np
import pickle
import time
import cv2
import os

# -- DIRECTORIES -- #
db_path = "../bbdd"
res_root = "../results"
qs1_w5 = "../qsd1_w5"

def main():
    global_start = time.time()

    print('-- READING IMAGES --')
    start = time.time()
    qs_paths = sorted(glob(qs1_w5+os.sep+'*.jpg'))
    qs_images = [cv2.imread(path) for path in qs_paths]
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- DENOISING IMAGES --')
    start = time.time()
    if not os.path.isfile(res_root+os.sep+'denoised.pkl'):
        denoiser = Denoiser(qs_images)
        qs_denoised = denoiser.denoise()
        with open(res_root+os.sep+'denoised.pkl','wb') as ff:
            pickle.dump(qs_denoised,ff)
    else:
        with open(res_root+os.sep+'denoised.pkl','rb') as ff:
            qs_denoised = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- DETECTING ORIENTATION --')
    start = time.time()
    if not (os.path.isfile(res_root+os.sep+'angles.pkl') and os.path.isfile(res_root+os.sep+'rotated.pkl')):
        orientation = Orientation(qs_denoised)
        qs_angles, qs_rotated = orientation.compute_orientation()
        with open(res_root+os.sep+'angles.pkl','wb') as ff:
            pickle.dump(qs_angles,ff)
        with open(res_root+os.sep+'rotated.pkl','wb') as ff:
            pickle.dump(qs_rotated,ff)
    else:
        with open(res_root+os.sep+'angles.pkl','rb') as ff:
            qs_angles = pickle.load(ff)
        with open(res_root+os.sep+'rotated.pkl','rb') as ff:
            qs_rotated = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- EVALUATING ANGLES --')
    start = time.time()
    evaluator = EvaluateAngles(qs_angles,qs1_w5+os.sep+'angles_qsd1w5.pkl')
    score = evaluator.evaluate(degree_margin=1)
    score = evaluator.evaluate(degree_margin=5)
    print('-- DONE: Time: '+str(time.time()-start))

if __name__ == "__main__":
    main()
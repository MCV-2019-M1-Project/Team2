# -- SCRIPT TO FIND THE BEST DESCRIPTORS AND TEST VARIOUS THINGS -- #

# -- IMPORTS -- #
from noise import Denoiser
from detect_orientation import Orientation
from paintings_count import SplitImages
from background_removal import BackgroundRemoval, GetForegroundPixels
from textbox_removal import TextDetection
from evaluation import EvaluateAngles, EvaluateIoU
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

def main(eval_=True):
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

    if eval_:
        print('-- EVALUATING ANGLES --')
        start = time.time()
        angle_evaluator = EvaluateAngles(qs_angles,qs1_w5+os.sep+'angles_qsd1w5.pkl')
        score = angle_evaluator.evaluate(degree_margin=5)
        print('-- DONE: Time: '+str(time.time()-start))

    print('-- SPLITTING IMAGES --')
    start = time.time()
    if not os.path.isfile(res_root+os.sep+'splitted.pkl'):
        spliter = SplitImages(qs_rotated)
        qs_splitted = spliter.get_paintings()
        with open(res_root+os.sep+'splitted.pkl','wb') as ff:
            pickle.dump(qs_splitted,ff)
    else:
        with open(res_root+os.sep+'splitted.pkl','rb') as ff:
            qs_splitted = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTE FOREGROUND --')
    start = time.time()
    if not os.path.isfile(res_root+os.sep+'qs_masks.pkl'):
        removal = BackgroundRemoval(qs_splitted)
        qs_masks = removal.remove_background()
        with open(res_root+os.sep+'qs_masks.pkl','wb') as ff:
            pickle.dump(qs_masks,ff)
    else:
        with open(res_root+os.sep+'qs_masks.pkl','rb') as ff:
            qs_masks = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTE TEXTBOXES --')
    start = time.time()
    if not (os.path.isfile(res_root+os.sep+'text_mask.pkl') and os.path.isfile(res_root+os.sep+'text_boxes.pkl')):
        text_removal = TextDetection(qs_splitted)
        text_masks, text_boxes = text_removal.detect()
        with open(res_root+os.sep+'text_boxes.pkl','wb') as ff:
            pickle.dump(text_boxes,ff)
        with open(res_root+os.sep+'text_masks.pkl','wb') as ff:
            pickle.dump(text_masks,ff)
    else:
        with open(res_root+os.sep+'text_boxes.pkl','rb') as ff:
            text_boxes = pickle.load(ff)
        with open(res_root+os.sep+'text_masks.pkl','rb') as ff:
            text_masks = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- GETTING FG PIXELS --')
    start = time.time()
    num = 0
    for img,mask in zip(qs_splitted,qs_masks):
        for paint,pmask in zip(img,mask):
            fg = GetForegroundPixels(paint,mask)
            cv2.imwrite('../results/Foreground/{0:02}.png',fg)
            num += 1
    print('-- DONE: Time: '+str(time.time()-start))

if __name__ == "__main__":
    main(eval_=True)
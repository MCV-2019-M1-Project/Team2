# -- SCRIPT TO FIND THE BEST DESCRIPTORS AND TEST VARIOUS THINGS -- #

# -- IMPORTS -- #
from skimage import restoration as R
from glob import glob
import numpy as np
import time
import cv2
import os

# -- DIRECTORIES -- #
db_path = "../bbdd"
res_root = "../results"
qs1_w5 = "../qsd1_w5"

def test_noise():
    start = time.time()
    # -- TEST REDUCE NOISE -- #
    qs_paths = sorted(glob(qs1_w5+os.sep+'*.jpg'))
    qs_images = [cv2.imread(qs_paths[k]) for k in [7,8,13,15,16,20,26]]

    directory = res_root+os.sep+'Denoised'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    for i,img in enumerate(qs_images):
        img = cv2.medianBlur(img,ksize=3)
        img = cv2.fastNlMeansDenoisingColored(img,h=7,hColor=3,templateWindowSize=7,searchWindowSize=21)
        cv2.imwrite(directory+os.sep+'{0:03d}.png'.format(i),img)

    print("Done. Time: " + str(time.time() - start))

def test_boxes():
    start = time.time()
    # -- TEST DETECT BOXES -- #
    qs_paths = sorted(glob(qs1_w5+os.sep+'*.jpg'))
    qs_images = [cv2.imread(qs_paths[k]) for k in [1,2,6,29]]

    directory = res_root+os.sep+'Boxes'
    if not os.path.isdir(directory):
        os.mkdir(directory)

    for k,img in enumerate(qs_images):
        img = cv2.medianBlur(img,ksize=3)
        img = cv2.fastNlMeansDenoisingColored(img,h=7,hColor=3,templateWindowSize=7,searchWindowSize=21)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,10))
        dark = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        bright = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
        dark = np.abs(np.max(dark,axis=2)-np.min(dark,axis=2))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        bright = np.abs(np.max(bright,axis=2)-np.min(bright,axis=2))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        activity = bright+dark
        grad = cv2.morphologyEx(activity, cv2.MORPH_GRADIENT, kernel3)
        cv2.imwrite(directory+os.sep+'{0:03d}_grad.png'.format(k),grad)


if __name__ == "__main__":
    #test_noise()
    test_boxes()
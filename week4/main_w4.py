
# -- IMPORTS -- #
from glob import glob
from descriptor import ORBDescriptor, ShiThomasDescriptor, HarrisDescriptor, SIFTDescriptor, SURFDescriptor
from paintings_count import getListOfPaintings
from background_removal import BackgroundMask4
from textbox_removal import TextBoxRemoval
from evaluation import EvaluateDescriptors
from matchers import Matcher
from noise import Denoise
import numpy as np
import pickle
import cv2
import os
import time
import sys

# -- DIRECTORIES -- #
db_path = "../bbdd"
qs_path = "../qsd1_w4"
res_root = "../results"
qs_corresps_path = qs_path + "/gt_corresps.pkl"

# -- MAIN FOR QS1W4 -- #
def main_qs4(save=True):
    global_start = time.time()
    print('-- DENOISING IMAGES --')
    start = time.time()
    denoiser = Denoise(qs_path)
    denoiser.median_filter(3)
    qs_images = denoiser.tv_bregman(weight=0.01,max_iter=1000,eps=0.001,isotropic=True)
    qs_images = [[cv2.resize(item[0],(1000,1000))] for item in qs_images]
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- GETTING IMAGES --')
    start = time.time()
    qs_images = getListOfPaintings(qs_images,"EDGES")
    if save:
        with open(res_root+os.sep+'qs_images.pkl','wb') as ff:
            pickle.dump(qs_images,ff)
    db_images = []
    for path in sorted(glob(db_path+os.sep+'*.jpg')):
        db_images.append([cv2.imread(path)])
    db_images = [[cv2.resize(item[0],(1000,1000))] for item in db_images]
    if save:
        with open(res_root+os.sep+'db_images.pkl','wb') as ff:
            pickle.dump(db_images,ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- GETTING MASKS --')
    start = time.time()
    qs_masks = []
    for ind,img in enumerate(qs_images):
        qs_masks.append([])
        for paint in img:
            mask, mean_points = BackgroundMask4(paint)
            qs_masks[-1].append(mask)
            """UNCOMMENT LINE TO PRODUCE THE MASK TO UPLOAD TO THE SERVER"""
        cv2.imwrite(res_root+os.sep+"QS1W4/{0:05d}.png".format(ind),np.concatenate([item for item in qs_masks[-1]],axis=1))
    if save:
        with open(res_root+os.sep+'qs_masks.pkl','wb') as ff:
            pickle.dump(qs_masks,ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- GETTING TEXTBOXES --')
    start = time.time()
    qs_bbox = []
    for ind,img in enumerate(qs_images):
        qs_bbox.append([])
        for p, paint in enumerate(img):
            painting_masked = cv2.bitwise_and(paint,paint,mask=qs_masks[ind][p])
            cv2.imwrite(res_root+os.sep+'{0:05d}.png'.format(p),painting_masked)
            _, textbox = TextBoxRemoval(painting_masked)
            qs_bbox[-1].append([textbox[0][1],textbox[0][0],textbox[1][1],textbox[1][0]])
    if save:
        with open(res_root+os.sep+'qs_bbox.pkl','wb') as ff:
            pickle.dump(qs_bbox,ff)
    print('-- DONE: Time: '+str(time.time()-start))

def test_only_desc():
    global_start = time.time()
    print('-- GETTING INFO --')
    start = time.time()
    with open(res_root+os.sep+'qs_images.pkl','rb') as ff:
        qs_images = pickle.load(ff)
    with open(res_root+os.sep+'db_images.pkl','rb') as ff:
        db_images = pickle.load(ff)
    with open(res_root+os.sep+'qs_masks.pkl','rb') as ff:
        qs_masks = pickle.load(ff)
    with open(res_root+os.sep+'qs_bbox.pkl','rb') as ff:
        qs_bbox = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTING QUERY DESCRIPTORS --')
    start = time.time()
    qs_desc = ShiThomasDescriptor(qs_images,qs_masks,qs_bbox)
    qs_desc.compute_descriptors()
    with open(res_root+os.sep+'qs_result.pkl','wb') as ff:
        pickle.dump(qs_desc.result,ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTING BBDD DESCRIPTORS --')
    start = time.time()
    db_desc = ShiThomasDescriptor(db_images)
    db_desc.compute_descriptors()
    with open(res_root+os.sep+'db_result.pkl','wb') as ff:
        pickle.dump(db_desc.result,ff)
    print('-- DONE: Time: '+str(time.time()-start))
    
    print('-- MATCHING --')
    start = time.time()
    matcher = Matcher(db_desc.result, qs_desc.result, measure=cv2.NORM_HAMMING)
    matcher.match()
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- EVALUATING --')
    start = time.time()
    evaluator = EvaluateDescriptors(matcher.result,qs_corresps_path)
    evaluator.compute_mapatk(limit=1)
    print("MAP@1 = "+str(evaluator.score))
    evaluator.compute_mapatk(limit=5)
    print("MAP@1 = "+str(evaluator.score))
    print("Done. Time: "+str(time.time()-start))

    print("Total time: "+str(time.time()-global_start))

if __name__ == "__main__":
    test_only_desc()    
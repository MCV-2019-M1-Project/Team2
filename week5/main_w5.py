# -- SCRIPT TO FIND THE BEST DESCRIPTORS AND TEST VARIOUS THINGS -- #

# -- IMPORTS -- #
from noise import Denoiser
from detect_orientation import Orientation, Unrotate
from paintings_count import SplitImages
from background_removal import BackgroundRemoval, GetForegroundPixels
from text_detection import TextDetection
from evaluation import EvaluateAngles, EvaluateIoU, EvaluateDescriptors
from descriptor import SIFTDescriptor, ORBDescriptor
from matchers import MatcherFLANN, MatcherBF
from glob import glob
import numpy as np
import math
import pickle
import time
import cv2
import os

# -- DIRECTORIES -- #
db_path = "../bbdd"
res_root = "../results"
qs1_w5 = "../qst1_w5"

def main(eval_=True):
    global_start = time.time()

    print('-- READING IMAGES --')
    start = time.time()
    db_paths = sorted(glob(db_path+os.sep+'*.jpg'))
    qs_paths = sorted(glob(qs1_w5+os.sep+'*.jpg'))
    db_images = [[cv2.imread(path)] for path in db_paths]
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
    if not (os.path.isfile(res_root+os.sep+'angles.pkl') and os.path.isfile(res_root+os.sep+'rotated.pkl') and os.path.isfile(res_root+os.sep+'angles_real.pkl')):
        orientation = Orientation(qs_denoised)
        qs_angles, qs_angles_real, qs_rotated = orientation.compute_orientation()
        with open(res_root+os.sep+'angles.pkl','wb') as ff:
            pickle.dump(qs_angles,ff)
        with open(res_root+os.sep+'angles_real.pkl','wb') as ff:
            pickle.dump(qs_angles_real,ff)
        with open(res_root+os.sep+'rotated.pkl','wb') as ff:
            pickle.dump(qs_rotated,ff)
    else:
        with open(res_root+os.sep+'angles.pkl','rb') as ff:
            qs_angles = pickle.load(ff)
        with open(res_root+os.sep+'angles_real.pkl','rb') as ff:
            qs_angles_real = pickle.load(ff)
        with open(res_root+os.sep+'rotated.pkl','rb') as ff:
            qs_rotated = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    if eval_:
        print('-- EVALUATING ANGLES --')
        start = time.time()
        angle_evaluator = EvaluateAngles(qs_angles,qs1_w5+os.sep+'angles_qsd1w5.pkl')
        score = angle_evaluator.evaluate(degree_margin=1.5)
        print('-- DONE: Time: '+str(time.time()-start))

    print('-- SPLITTING IMAGES --')
    start = time.time()
    if not (os.path.isfile(res_root+os.sep+'splitted.pkl') and os.path.isfile(res_root+os.sep+'qs_displays.pkl')):
        spliter = SplitImages(qs_rotated)
        qs_splitted, qs_displays = spliter.get_paintings()
        with open(res_root+os.sep+'splitted.pkl','wb') as ff:
            pickle.dump(qs_splitted,ff)
        with open(res_root+os.sep+'qs_displays.pkl','wb') as ff:
            pickle.dump(qs_displays,ff)
    else:
        with open(res_root+os.sep+'splitted.pkl','rb') as ff:
            qs_splitted = pickle.load(ff)
        with open(res_root+os.sep+'qs_displays.pkl','rb') as ff:
            qs_displays = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTE FOREGROUND --')
    start = time.time()
    if not (os.path.isfile(res_root+os.sep+'qs_masks_rot.pkl') and os.path.isfile(res_root+os.sep+'qs_bboxs_rot.pkl')):
        removal = BackgroundRemoval(qs_splitted)
        qs_masks_rot, qs_bboxs_rot = removal.remove_background()
        with open(res_root+os.sep+'qs_masks_rot.pkl','wb') as ff:
            pickle.dump(qs_masks_rot,ff)
        with open(res_root+os.sep+'qs_bboxs_rot.pkl','wb') as ff:
            pickle.dump(qs_bboxs_rot,ff)
    else:
        with open(res_root+os.sep+'qs_masks_rot.pkl','rb') as ff:
            qs_masks_rot = pickle.load(ff)
        with open(res_root+os.sep+'qs_bboxs_rot.pkl','rb') as ff:
            qs_bboxs_rot = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- UNROTATE MASKS AND FOREGROUND BOUNDING BOXES --')
    start = time.time()
    if not (os.path.isfile(res_root+os.sep+'qs_masks.pkl') and os.path.isfile(res_root+os.sep+'qs_bboxs.pkl')):
        undo_rotation = Unrotate(qs_images)
        qs_masks, qs_bboxs = undo_rotation.unrotate(qs_angles,qs_bboxs_rot,qs_masks_rot,qs_displays)
        with open(res_root+os.sep+'qs_masks.pkl','wb') as ff:
            pickle.dump(qs_masks,ff)
        with open(res_root+os.sep+'qs_bboxs.pkl','wb') as ff:
            pickle.dump(qs_bboxs,ff)
    else:
        with open(res_root+os.sep+'qs_masks.pkl','rb') as ff:
            qs_masks = pickle.load(ff)
        with open(res_root+os.sep+'qs_bboxs.pkl','rb') as ff:
            qs_bboxs = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTE FRAMES OUTPUT PICKLE --')
    start = time.time()
    if not os.path.isfile(res_root+os.sep+'frames.pkl'):
        qs_frames = []
        for ind,bboxs in enumerate(qs_bboxs):
            qs_frames.append([])
            for ind2,bbox in enumerate(bboxs):
                qs_frames[-1].append([qs_angles[ind],bbox])
        with open(res_root+os.sep+'frames.pkl','wb') as ff:
            pickle.dump(qs_frames,ff)
    else:
        with open(res_root+os.sep+'frames.pkl','rb') as ff:
            qs_frames = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))    

    print('-- COMPUTE TEXTBOXES --')
    start = time.time()
    if not os.path.isfile(res_root+os.sep+'text_masks.pkl'):
        text_removal = TextDetection(qs_splitted)
        text_masks = text_removal.detect()
        with open(res_root+os.sep+'text_masks.pkl','wb') as ff:
            pickle.dump(text_masks,ff)
    else:
        with open(res_root+os.sep+'text_masks.pkl','rb') as ff:
            text_masks = pickle.load(ff)
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTE DESCRIPTORS --')
    start = time.time()
    #db_desc = SIFTDescriptor(db_images,None,None)
    #qs_desc = SIFTDescriptor(qs_splitted,mask_list=qs_masks_rot,bbox_list=text_masks)
    db_desc = ORBDescriptor(db_images,None,None)
    qs_desc = ORBDescriptor(qs_splitted,mask_list=qs_masks_rot,bbox_list=text_masks)
    db_desc.compute_descriptors()
    qs_desc.compute_descriptors()
    print('-- DONE: Time: '+str(time.time()-start))

    print('-- COMPUTE MATCHES --')
    start = time.time()
    matcher = MatcherFLANN(db_desc.result,qs_desc.result,flag=True)
    matcher.match(min_matches=12,match_ratio=0.65)
    with open('../results/result.pkl','wb') as ff:
        pickle.dump(matcher.result,ff)
    print('-- DONE: Time: '+str(time.time()-start))

    if eval_:
        print('-- EVALUATING DESCRIPTORS --')
        start = time.time()
        desc_evaluator = EvaluateDescriptors(matcher.result,qs1_w5+os.sep+'gt_corresps.pkl')
        desc_evaluator.compute_mapatk(limit=1)
        print('MAP@1: [{0}]'.format(desc_evaluator.score))
        desc_evaluator.compute_mapatk(limit=5)
        print('MAP@5: [{0}]'.format(desc_evaluator.score))
    print('-- Total time: '+str(time.time()-global_start))

if __name__ == "__main__":
    main(eval_=False)

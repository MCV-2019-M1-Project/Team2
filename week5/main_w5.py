# -- SCRIPT TO FIND THE BEST DESCRIPTORS AND TEST VARIOUS THINGS -- #

# -- IMPORTS -- #
from noise import Denoiser
from detect_orientation import Orientation
from paintings_count import SplitImages
from background_removal import BackgroundRemoval, GetForegroundPixels
from textbox_removal import TextDetection
from evaluation import EvaluateAngles, EvaluateIoU
from glob import glob
from scipy.ndimage import rotate
import numpy as np
import math
import pickle
import time
import cv2
import os

# This function has to go somewhere
def rotate_point(x,y,xm,ym,a):
    a = a*math.pi/180
    xr = (x - xm) * math.cos(a) - (y - ym) * math.sin(a) + xm
    yr = (x - xm) * math.sin(a) + (y - ym) * math.cos(a) + ym
    return [int(xr), int(yr)]
#

# -- DIRECTORIES -- #
db_path = r"C:\Users\PC\Documents\Roger\Master\M1\Project\bbdd"
res_root = "../results"
qs1_w5 = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5"

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
        score = angle_evaluator.evaluate(degree_margin=5)
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
        ### --- Això ha d'anar a algun lloc ---
        qs_masks = []
        qs_bboxs = []
        for ind,qs_img in enumerate(qs_images):
            # Variables needed
            angle_real = qs_angles_real[ind]
            split_masks_bboxs = qs_bboxs_rot[ind]
            split_masks = qs_masks_rot[ind]
            display = qs_displays[ind]
            axis = 0 if display == "vertical" else 1

            # Join masks in 1 image
            join_mask = np.concatenate(split_masks,axis=axis)
            # Rotate joined mask without reshaping
            join_mask_unrot = rotate(join_mask,-angle_real,reshape=False)

            # Compute cutting points for returning to original image
            to_cut = (np.array(join_mask_unrot.shape[:2]) - np.array(qs_img.shape[:2]))/2
            # Cut rotated joined mask
            join_mask_unrot_cut = join_mask_unrot[int(to_cut[0]):int(join_mask_unrot.shape[0]-to_cut[0]),
                                                  int(to_cut[1]):int(join_mask_unrot.shape[1]-to_cut[1])]
            # Append final mask for qs_img to result list
            qs_masks.append(join_mask_unrot_cut)

            # Adapt foreground bboxes according to image size
            join_bboxs = [split_masks_bboxs[0]]
            for ind,item in enumerate(split_masks_bboxs):
                if ind == 0:
                    continue
                add_shape_x = 0
                add_shape_y = 0
                if display == "vertical":
                    add_shape_x = np.sum([subitem.shape[0] for subitem in split_masks[:ind]])
                else:
                    add_shape_y = np.sum([subitem.shape[1] for subitem in split_masks[:ind]])
                join_bboxs.append([[subitem[0]+add_shape_x,subitem[1]+add_shape_y] for subitem in item])

            # Compute central point for rotation
            central_point = [int(join_mask_unrot.shape[0]/2),int(join_mask_unrot.shape[1]/2)]
            # Rotate bbox points to original orientation
            join_bboxs_unrot = []
            for item in join_bboxs:
                to_append = []
                for point in item:
                    rotated_point = rotate_point(point[0],point[1],central_point[0],central_point[1],-angle_real)
                    rotated_point_cut = [int(rotated_point[0]-to_cut[0]),int(rotated_point[1]-to_cut[1])]
                    to_append.append(rotated_point_cut)
                join_bboxs_unrot.append(to_append)
            # Append final foreground bboxes for qs_img to result_list
            qs_bboxs.append(join_bboxs_unrot)
        ### --- ---
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
    if not (os.path.isfile(res_root+os.sep+'text_masks.pkl') and os.path.isfile(res_root+os.sep+'text_boxes.pkl')):
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
    
    print('-- Total time: '+str(time.time()-global_start))

    
if __name__ == "__main__":
    main(eval_=True)

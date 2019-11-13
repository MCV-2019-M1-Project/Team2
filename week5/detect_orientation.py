# -- IMPORTS -- #
from scipy import ndimage
from glob import glob
import numpy as np
import cv2
import os
import math

class Orientation():
    """CLASS::Orientation:
        >- Detects the orientation of all the paintings using HoughLines."""
    def __init__(self, img_list):
        self.img_list = img_list

    def compute_orientation(self):
        result = [self.detect_orientation(item,k) for k,item in enumerate(self.img_list)]
        self.angles = [item[0] for item in result]
        self.angles_real = [item[1] for item in result]
        self.rotated_img = [item[2] for item in result]
        return self.angles, self.angles_real, self.rotated_img

    def detect_orientation(self,img,k):
        # Converting to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Computing edges
        edges = cv2.Canny(gray, 100, 100, apertureSize=3)
        # Computing HoguhLines
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=20)
        mark = np.copy(img)
        angles = []
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(mark,(x1,y1),(x2,y2),(0,0,255),3)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        #Line to generate image for slides 
        cv2.imwrite('../results/Orientation/{0:02}_lines.png'.format(k),mark)

        median_angle = np.median(angles)
        original_median_angle = median_angle
        if median_angle > 45:
            median_angle = abs(median_angle)-90
        elif median_angle < -45:
            median_angle = median_angle+90
        img_rotated = ndimage.rotate(img, median_angle)
        rotated_angle = median_angle

        inpainting_mask = np.ones(shape=[img.shape[0],img.shape[1]],dtype=np.uint8)
        inpainting_mask = (1-ndimage.rotate(inpainting_mask,median_angle))
        img_inpaint = cv2.inpaint(img_rotated,inpainting_mask,3,cv2.INPAINT_TELEA)
        img_inpaint[:,0] = img_inpaint[:,1]
        img_inpaint[:,-1] = img_inpaint[:,-2]
        img_inpaint[0,:] = img_inpaint[1,:]
        img_inpaint[-1,:] = img_inpaint[-2,:]

        #Line to generate image for slides 
        mark = np.copy(img_inpaint)
        cv2.imwrite('../results/Orientation/{0:02}_final.png'.format(k), mark)

        if median_angle > 0:
                median_angle = 180 - median_angle
        else:
            median_angle = np.abs(median_angle)
        print('Image ['+str(k)+'] Processed.')
        return median_angle, rotated_angle, img_inpaint

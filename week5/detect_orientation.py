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
        #cv2.imwrite('../results/Orientation/{0:02}_lines.png'.format(k),mark)

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

class Unrotate():
    """CLASS::Unrotate:
        >- Unrotates the fg masks and splitted images"""
    def __init__(self,qs_images):
        self.qs_images = qs_images
        self.qs_masks = []
        self.qs_bboxs = []

    def unrotate(self,qs_angles,qs_bboxs_rot,qs_masks_rot,qs_displays):
        for ind,qs_img in enumerate(self.qs_images):
            # Variables needed
            angle_real = qs_angles[ind]
            split_masks_bboxs = qs_bboxs_rot[ind]
            split_masks = qs_masks_rot[ind]
            display = qs_displays[ind]
            axis = 0 if display == "vertical" else 1

            # Join masks in 1 image
            join_mask = np.concatenate(split_masks,axis=axis)
            # Rotate joined mask without reshaping
            join_mask_unrot = ndimage.rotate(join_mask,-angle_real,reshape=False)

            # Compute cutting points for returning to original image
            to_cut = (np.array(join_mask_unrot.shape[:2]) - np.array(qs_img.shape[:2]))/2
            # Cut rotated joined mask
            join_mask_unrot_cut = join_mask_unrot[int(to_cut[0]):int(join_mask_unrot.shape[0]-to_cut[0]),
                                                  int(to_cut[1]):int(join_mask_unrot.shape[1]-to_cut[1])]
            # Append final mask for qs_img to result list
            self.qs_masks.append(join_mask_unrot_cut)

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
                    rotated_point = self.rotate_point(point[0],point[1],central_point[0],central_point[1],-angle_real)
                    rotated_point_cut = [int(rotated_point[0]-to_cut[0]),int(rotated_point[1]-to_cut[1])]
                    to_append.append(rotated_point_cut)
                join_bboxs_unrot.append(to_append)
            # Append final foreground bboxes for qs_img to result_list
            self.qs_bboxs.append(join_bboxs_unrot)
        return self.qs_masks, self.qs_bboxs

    def rotate_point(self,x,y,xm,ym,a):
        a = a*math.pi/180
        xr = (x - xm) * math.cos(a) - (y - ym) * math.sin(a) + xm
        yr = (x - xm) * math.sin(a) + (y - ym) * math.cos(a) + ym
        return [int(xr), int(yr)]
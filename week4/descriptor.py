import cv2
import numpy as np
from glob import glob
import os

class BaseDescriptor():
    """CLASS::BaseDescriptor:
        >- Class from which all corner descriptors form. """
    def __init__(self, img_list, mask_list, bbox_list):
        self.img_list = img_list
        self.mask_list = mask_list
        self.bbox_list = bbox_list
        self.result = {}
    
    def compute_descriptors(self):
        print('--- COMPUTING DESCRIPTORS --- ')
        for k, img in enumerate(self.img_list):
            self.result[k] = []
            print(str(k)+' out of '+str(len(self.img_list)))
            for i, paint in enumerate(img):
                pimg = self._apply_masks(paint, self.mask_list[k][i], self.bbox_list[k][i])
                self.result[k].append(self._compute_features(pimg))
        print('--- DONE --- ')
        return self.result
    def _apply_masks(self,paint,mask,bbox):
        """ IN THE CASE OF QS1_W4 ALL IMAGES HAVE BBOX AND MASK
            The mask should be the one that only counts the elimination of the background. """
        mask[bbox["top"]:bbox["bottom"],bbox["left"]:bbox["right"]] = 0
        return cv2.bitwise_and(paint,paint,mask=mask)
        
    def _compute_features(self,img,mask,bbox):
        """ SHOULD RETURN ONLY THE DESCRIPTORS, KEYPOINTS ARE JUST USEFUL TO COMPUTE THE DESCRIPTORS. """
        raise NotImplementedError('This is a base class, this will never be implemented.')

class HarrisDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list, bbox_list):
        super().__init__(img_list,mask_list,bbox_list)

    def _compute_features(self, img):
        thresh = 200
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blockSize = 2; apertureSize = 3; k = 0.04
        dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
        dst_norm = cv2.normalize(dst, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        keypoints = []
        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > thresh:
                    keypoints.append(cv2.KeyPoint(j, i, _size=3))
        sift = cv2.xfeatures2d.SIFT_create()
        keydescriptors = [sift.compute(gray, [kp])[1] for kp in keypoints]
        return np.array(keydescriptors)

class ORBDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list, bbox_list):
        super().__init__(img_list,mask_list,bbox_list)
        
    def _compute_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(gray,None)
        return des


import cv2
import numpy as np
import pickle
import time
import os

class BaseDescriptor():
    """CLASS::BaseDescriptor:
        >- Class from which all corner descriptors form. """
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        self.img_list = img_list
        self.mask_list = mask_list if mask_list else [[None]]*len(self.img_list)
        self.bbox_list = bbox_list if bbox_list else [[None]]*len(self.img_list)
        self.result = {}
    
    def compute_descriptors(self):
        print('--- COMPUTING DESCRIPTORS --- ')
        for k, img in enumerate(self.img_list):
            self.result[k] = []
            print(str(k)+' out of '+str(len(self.img_list)))
            for i, paint in enumerate(img):
                mask = self._combine_masks(self.mask_list[k][i], self.bbox_list[k][i])
                #cv2.imwrite('../results/Masks/{0:02}_{1}.png'.format(k,i),paint)
                self.result[k].append(self._compute_features(paint,mask))
        print('--- DONE --- ')

    def _combine_masks(self,foreground,textbox):
        """ IN THE CASE OF QS1_W4 ALL IMAGES HAVE BBOX AND MASK
            The mask should be the one that only counts the elimination of the background. """
        if foreground is not None:
            if textbox is not None:
                mask = cv2.bitwise_and(foreground,foreground,mask=textbox)
            else:
                mask = foreground
            return mask
        else:
            return None
        
    def _compute_features(self,img,mask,bbox):
        raise NotImplementedError('This is a base class, this will never be implemented.')

class ORBDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)
        """ IF WTA_K IS GREATER THAN 2 DISTANCE ON MATHCER NEEDS TO BE HAMMING2 OTHERWISE HAMMING"""
        nfeatures = 1500; scaleFactor = 1.2; nlevels = 8; edgeThreshold = 31; firstLevel = 0; wta_k = 2
        self.orb = cv2.ORB_create(nfeatures=nfeatures,
                                scaleFactor=scaleFactor,
                                nlevels=nlevels,
                                edgeThreshold=edgeThreshold,
                                firstLevel=firstLevel,
                                WTA_K=wta_k)
        
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        gray = cv2.resize(gray,(512,512))
        mask = cv2.resize(mask,(512,512))
        kkpp, des = self.orb.detectAndCompute(gray,mask,None)
        kp = np.array([k.pt for k in kkpp])
        return kp,des

class SIFTDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)
        nfeatures = 1000; nOctaveLayers = 3
        contrastThreshold = 0.06; edgeThreshold = 31; sigma = 1.6
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma)
    
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        gray = cv2.resize(gray,(512,512))
        mask = cv2.resize(mask,(512,512))
        kkpp, desc = self.sift.detectAndCompute(gray,mask,None)
        kp = np.array([k.pt for k in kkpp])
        return kp,desc

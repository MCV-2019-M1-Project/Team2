import cv2
import numpy as np
from glob import glob
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
                self.result[k].append(self._compute_features(paint,mask))
        print('--- DONE --- ')

    def _combine_masks(self,mask,bbox):
        """ IN THE CASE OF QS1_W4 ALL IMAGES HAVE BBOX AND MASK
            The mask should be the one that only counts the elimination of the background. """
        if mask is not None:
            if bbox is not None:
                mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 0
            return mask
        else:
            return None
        
    def _compute_features(self,img,mask,bbox):
        raise NotImplementedError('This is a base class, this will never be implemented.')

class HarrisDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)

    def _compute_features(self, img, mask):
        thresh = 200
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        blockSize = 2; apertureSize = 3; k = 0.04
        keypoints = cv2.cornerHarris(gray, blockSize, apertureSize, k)
        kkpp = []
        for point in keypoints:
            x,y = point.ravel()
            kkpp.append(cv2.KeyPoint(x, y, _size=3))
        sift = cv2.xfeatures2d.SIFT_create()
        _,des = sift.compute(gray, kkpp)
        kp = np.array([k.pt for k in kkpp])
        return kp,des

class ORBDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)
        """ IF WTA_K IS GREATER THAN 2 DISTANCE ON MATHCER NEEDS TO BE HAMMING2 OTHERWISE HAMMING"""
        nfeatures = 3000; scaleFactor = 1.2; nlevels = 8; edgeThreshold = 31; firstLevel = 0; wta_k = 2
        self.orb = cv2.ORB_create(nfeatures=nfeatures,
                                scaleFactor=scaleFactor,
                                nlevels=nlevels,
                                edgeThreshold=edgeThreshold,
                                firstLevel=firstLevel,
                                WTA_K=wta_k)
        
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        gray = cv2.resize(gray,(256,256))
        mask = cv2.resize(mask,(256,256))
        kkpp, des = self.orb.detectAndCompute(gray,mask,None)
        kp = np.array([k.pt for k in kkpp])
        return kp,des

class ShiThomasDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list,mask_list,bbox_list)
        self.num = 0
        hessianThreshold = 400; nOctaves = 4; nOctaveLayers = 3
        extended = False; upright = False
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
    
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        maxCorners=50; qualityLevel=0.01; minDistance=25; blockSize = 3
        flag = True if mask is None else False
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        keypoints = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, blockSize, mask=mask)
        if keypoints is None:
            return (None,None)
        marker = img
        kkpp = []
        for point in keypoints:
            x,y = point.ravel()
            kkpp.append(cv2.KeyPoint(x, y, _size=3))
            cv2.circle(marker, (x,y), 10, 152, -1)
        if flag:
            cv2.imwrite('../results/DB_SHI/{0:05d}.png'.format(self.num),marker)
        else:
            cv2.imwrite('../results/MASKS/{0:05d}.png'.format(self.num),mask)
            cv2.imwrite('../results/SHI/{0:05d}.png'.format(self.num),marker)
        _,des = self.surf.compute(gray,kkpp)
        kp = np.array([k.pt for k in kkpp])
        self.num +=1
        return kp,des

class SIFTDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)
        nfeatures = 0; nOctaveLayers = 3,
        contrastThreshold = 0.04; edgeThreshold = 10; sigma = 1.6
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures,
                        nOctaveLayers,
                        contrastThreshold,
                        edgeThreshold,
                        sigma)
    
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        kkpp, desc = self.sift.detectAndCompute(gray,mask,None)
        kp = np.array([k.pt for k in kkpp])
        return kp,desc

class SURFDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)
        hessianThreshold = 400; nOctaves = 4; nOctaveLayers = 2; extended = False; upright = True
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
        # self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, upright=True)
    
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        kkpp, des = self.surf.detectAndCompute(gray,mask,None)
        kp = np.array([k.pt for k in kkpp])
        return kp,des

class DAISYDescriptor(BaseDescriptor):
    def __init__(self, img_list, mask_list=None, bbox_list=None):
        super().__init__(img_list, mask_list, bbox_list)
        self.daisy = cv2.xfeatures2d.DAISY_create()
    
    def _compute_features(self, img, mask):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = np.array(mask,dtype=np.uint8) if mask is not None else np.ones_like(gray,dtype=np.uint8)*255
        kkpp, des = self.daisy.detectAndCompute(gray,mask,None)
        kp = np.array([k.pt for k in kkpp])
        return kp,des

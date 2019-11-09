# -- IMPORTS -- #
from skimage import restoration as R
from glob import glob
import numpy as np
import cv2
import os

class Denoiser():
    """CLASS::Denoiser:
        >- Responsible to apply median filter and NonLocalMeans 
            >- Information about NonLocalMeans: https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf"""
    def __init__(self, img_list):
        self.img_list = img_list
    
    def denoise(self):
        return [self.denoise_img(img,k) for k,img in enumerate(self.img_list)]
    
    def denoise_img(self, img, k):
        # Median Filter
        img = cv2.medianBlur(img, ksize=3)
        # Non Local Means
        img = cv2.fastNlMeansDenoisingColored(img, h=7, hColor=3, templateWindowSize=7, searchWindowSize=21)
        print('Image ['+str(k)+'] Processed.')
        return img
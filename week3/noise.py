# -- SCRIPT FOR DENOISING-- #

# -- IMPORTS -- #
from skimage import restoration as R
from glob import glob
import numpy as np
import cv2
import os

""" Look at the following links for more information about the parameters:
    https://scikit-image.org/docs/dev/api/skimage.restoration.html
    https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga03aa4189fc3e31dafd638d90de335617
"""

# -- DENOISE -- #
class Denoise():
    """CLASS:Denoise:
        >- Class that implements all kinds of denoising techniques."""
    def __init__(self,img_root):
        img_paths = sorted(glob(img_root+os.sep+'*.jpg'))
        self.img = []
        for path in img_paths:
            self.img.append(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB))
    
    def bilateral(self,win_size=None,sigma_spatial=1,bins=1000,mode='constant'):
        denoised = [R.denoise_bilateral(np.array(item,np.float32),
                                    win_size=win_size,
                                    sigma_color=None,
                                    sigma_spatial=sigma_spatial,
                                    bins=bins,
                                    mode=mode,
                                    multichannel=True) for item in self.img]
        return [[cv2.cvtColor(np.array(item,np.uint8),cv2.COLOR_RGB2BGR)] for item in denoised]
    
    def nl_means(self,patch_size=7,patch_distance=11,cut_off=0.1,fast_mode=True,sigma=0.0):
        denoised = [R.denoise_nl_means(np.array(item,np.float32),
                                    patch_size=patch_size,
                                    patch_distance=patch_distance,
                                    h=cut_off,
                                    fast_mode=fast_mode,
                                    sigma=sigma) for item in self.img]
        return [[cv2.cvtColor(np.array(item,np.uint8),cv2.COLOR_RGB2BGR)] for item in denoised]

    def tv_bregman(self,weight,max_iter=100,eps=0.001,isotropic=True):
        denoised = [R.denoise_tv_bregman(np.array(item,np.float32),
                                    weight=weight,
                                    max_iter=max_iter,
                                    eps=eps,
                                    isotropic=isotropic) for item in self.img]
        return [[cv2.cvtColor(np.array(item,np.uint8),cv2.COLOR_RGB2BGR)] for item in denoised]

    def tv_chambolle(self,weight=0.1,eps=0.0002,max_iter=200):
        denoised = [R.denoise_tv_chambolle(np.array(item,np.float32),
                                        weight=weight,
                                        eps=eps,
                                        n_iter_max=max_iter,
                                        multichannel=True) for item in self.img]
        return [[cv2.cvtColor(np.array(item,np.uint8),cv2.COLOR_RGB2BGR)] for item in denoised]

    def wavelet(self,sigma=None,wavelet='db1',mode='soft',wav_lev=None,method='BayesShrink'):
        denoised = [R.denoise_wavelet(np.array(item,np.float32),
                                sigma=sigma,
                                wavelet=wavelet,
                                mode=mode,
                                wavelet_levels=wav_lev,
                                convert2ycbcr=True,
                                method=method,
                                multichannel=True) for item in self.img]
        return [[cv2.cvtColor(np.array(item,np.float32),cv2.COLOR_RGB2BGR)] for item in denoised]

    def fast_nlmean(self,h=10,hC=10,template_size=7,win_size=21):
        denoised = [cv2.fastNlMeansDenoisingColored(item,
                                                h=h,
                                                hColor=hC,
                                                templateWindowSize=template_size,
                                                searchWindowSize=win_size) for item in self.img]
        return [[cv2.cvtColor(np.array(item,np.uint8),cv2.COLOR_RGB2BGR)] for item in denoised]

       
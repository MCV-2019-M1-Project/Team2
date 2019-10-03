# -- UTILITIES USED ACROSS THE PROJECT -- #

# -- IMPORTS -- #
import numpy as np
import cv2

# -- UTILITIES -- #

def compute_histogram(img,mask=None,color_space='ycrcb'):
    """FUNCTION::COMPUTE_HISTOGRAM
        >- Returns:  numpy array representing an the histogram of chrominance."""
    if color_space is 'ycrcb':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        histogram = cv2.calcHist([img],[1,2],mask,[256,256],[0,256,0,256])
    elif color_space is 'lab':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
        histogram = cv2.calcHist([img],[1,2],mask,[],[])
    else:
        raise NotImplementedError
    return histogram

def extract_features(histogram,norm='lmax',sub_factor=16):
    """FUNCTION::EXTRACT_FEATURES
        >- Returns: numpy array representing the extracted feature from it's histogram."""
    flat_hist = histogram.flatten()
    if norm is 'lmax':
        norm = np.max(flat_hist)
    elif norm is 'l2':
        norm = np.sqrt(np.sum(np.abs(np.power(flat_hist,2)),axis=-1))
    else:
        raise NotImplementedError
    norm_hist = flat_hist/norm
    feature = np.zeros_like(norm_hist[0:-1:sub_factor])
    for i,_ in enumerate(feature[0:-sub_factor]):
        feature[i]=np.mean(norm_hist[i*sub_factor:(i+1)*sub_factor])
    return feature

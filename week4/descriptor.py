import cv2
import numpy as np
from glob import glob
import os

class HarrisDescriptor:
    def __init__(self, img_list, mask_list, bbox_list):
        self.img_list = img_list
        self.mask_list = mask_list
        self.bbox_list = bbox_list
        self.result = {}

    def compute_descriptors(self):
        for k, images in enumerate(self.img_list):
           # print(str(k)+' out of '+str(len(self.img_list)))
           self.result[k] = []
           for i, paint in enumerate(images):
               self.result[k].append(self._compute_features(paint, None, None)) # self.mask_list[k][i], self.bbox_list[k][i]))
        print('--- DONE --- ')

    def _compute_features(self, img, mask, bbox):
        # apply mask and bbox
        thresh = 200
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detector parameters
        blockSize = 2
        apertureSize = 3
        k = 0.04
        # Detecting corners
        dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
        # Normalizing
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        keypoints = []
        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > thresh:
                    keypoints.append(cv2.KeyPoint(j, i, _size=3))

        sift = cv2.xfeatures2d.SIFT_create()
        key_descriptors = [sift.compute(gray, [kp])[1] for kp in keypoints]
        return key_descriptors


if __name__ == '__main__':
    folder = "../qsd1_w3"
    images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(folder, "*.jpg")))]

    descriptor = HarrisDescriptor(images, None, None)
    print(descriptor.compute_descriptors())
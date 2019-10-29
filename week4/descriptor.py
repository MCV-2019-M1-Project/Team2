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
        return self.result

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
        dst_norm = cv2.normalize(dst, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        keypoints = []
        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > thresh:
                    keypoints.append(cv2.KeyPoint(j, i, _size=3))
        sift = cv2.xfeatures2d.SIFT_create()
        keydescriptors = [sift.compute(gray, [kp])[1] for kp in keypoints]
        return keypoints, keydescriptors


if __name__ == '__main__':
    query_folder = "../qsd1_w4"
    query_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(query_folder, "*.jpg")))]
    query_descriptors = HarrisDescriptor(query_images, None, None)
    query_results = query_descriptors.compute_descriptors()
    print(len(query_results))
    print(type(query_results[0][0][1]))

    bbdd_folder = "../bbdd"
    bbdd_images = [[cv2.imread(item)] for item in sorted(glob(os.path.join(bbdd_folder, "*.jpg")))]
    bbdd_descriptor = HarrisDescriptor(bbdd_images, None, None)
    bbdd_results = bbdd_descriptor.compute_descriptors()
    print(len(bbdd_results))
    print(type(bbdd_results[0][0][1]))

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches1 = matcher.match(np.asarray(query_results[0][0][1]), np.asarray(bbdd_results[0][0][1]))
    print(type(matches1))
    matches1 = sorted(matches1, key = lambda x:x.distance)

    print(matches1)

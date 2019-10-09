import os
from glob import glob
import cv2
import numpy as np
from background_removal import BackgroundMask2

def computeMaskToCountPaingings(img_path):
    kernel_size = 8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    print("\nComputing mask with method 2 for img_path ", img_path)
    img = cv2.imread(img_path)
    orig_shape = img.shape
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    mask = BackgroundMask2(img)
    filtered_mask = mask
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    #filtered_mask = cv2.resize(filtered_mask, (orig_shape[1], orig_shape[0]))
    return filtered_mask

def countNumberPaintingsBasedOnMask(mask, img_path):
    print("\nCounting number of paintings in img_path ", img_path)


def main(img_folder):
    qs2_root = "../qsd2_w2"
    for img_path in sorted(glob(os.path.join(img_folder, "*.jpg")))[:]:
        mask = computeMaskToCountPaingings(img_path)
        countNumberPaintingsBasedOnMask(mask, img_path)

if __name__ == '__main__':
    main("../qsd2_w2")
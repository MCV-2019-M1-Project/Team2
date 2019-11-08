import os
from glob import glob
import cv2
import numpy as np
import math
from scipy import ndimage


def detect_orientation(img,img_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=20)

    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    print("Original angle:",median_angle)
    if median_angle > 45:
        median_angle = abs(median_angle)-90
    elif median_angle < -45:
        median_angle = median_angle+90
    img_rotated = ndimage.rotate(img, median_angle)

    # cv2.imwrite(img_path.replace(".jpg","_rot.png"), img_rotated)

    inpainting_mask = np.ones(shape=[img.shape[0],img.shape[1]],dtype=np.uint8)
    inpainting_mask = (1-ndimage.rotate(inpainting_mask,median_angle))
    cv2.imwrite(img_path.replace("qsd1_w5_denoised","qsd1_w5_denoised_rotated").replace(".jpg","_mask.png"), inpainting_mask*255)
    img_inpaint = cv2.inpaint(img_rotated,inpainting_mask,3,cv2.INPAINT_TELEA)
    img_inpaint[:,0] = img_inpaint[:,1]
    img_inpaint[:,-1] = img_inpaint[:,-2]
    img_inpaint[0,:] = img_inpaint[1,:]
    img_inpaint[-1,:] = img_inpaint[-2,:]

    cv2.imwrite(img_path.replace("qsd1_w5_denoised","qsd1_w5_denoised_rotated"), img_inpaint)

    return median_angle

if __name__ == '__main__':
    imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5_denoised"
    for img_path in sorted(glob(os.path.join(imgs_folder, "*.jpg")))[:]:
        img = cv2.imread(img_path)
        ori = detect_orientation(img,img_path)
        print("Angle is",ori,"for image",img_path)
        print()

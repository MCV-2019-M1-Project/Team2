import os
from glob import glob
import cv2
import numpy as np
import math
from scipy import ndimage
import pickle


def detect_orientation(img,img_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=20)

    angles = []
    for x1, y1, x2, y2 in lines[0]:
        # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # try:
        #     slope=(y2-y1)/(x2-x1)
        # except:
        #     slope = 90
        angles.append(angle)

    median_angle = np.median(angles)
    original_median_angle = median_angle
    # print("Original angle:",median_angle)
    if median_angle > 45:
        median_angle = abs(median_angle)-90
    elif median_angle < -45:
        median_angle = median_angle+90
    img_rotated = ndimage.rotate(img, median_angle)

    # cv2.imwrite(img_path.replace("qsd1_w5_denoised","qsd1_w5_denoised_rotated").replace(".jpg","_line.png"), img)

    inpainting_mask = np.ones(shape=[img.shape[0],img.shape[1]],dtype=np.uint8)
    inpainting_mask = (1-ndimage.rotate(inpainting_mask,median_angle))
    # cv2.imwrite(img_path.replace("qsd1_w5_denoised","qsd1_w5_denoised_rotated").replace(".jpg","_mask.png"), inpainting_mask*255)
    img_inpaint = cv2.inpaint(img_rotated,inpainting_mask,3,cv2.INPAINT_TELEA)
    img_inpaint[:,0] = img_inpaint[:,1]
    img_inpaint[:,-1] = img_inpaint[:,-2]
    img_inpaint[0,:] = img_inpaint[1,:]
    img_inpaint[-1,:] = img_inpaint[-2,:]

    # cv2.imwrite(img_path.replace("qsd1_w5_denoised","qsd1_w5_denoised_rotated"), img_inpaint)

    median_angle = np.abs(median_angle) if original_median_angle >= 0 else 180-np.abs(median_angle)

    return median_angle, img_inpaint

if __name__ == '__main__':
    imgs_folder = r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5_denoised"
    angles = []
    for img_path in sorted(glob(os.path.join(imgs_folder, "*.jpg")))[:]:
        img = cv2.imread(img_path)
        ori, rotated_img = detect_orientation(img,img_path)
        print("Angle is",ori,"for image",img_path)
        print()
        angles.append(ori)

    gt = pickle.load(open(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week5\qsd1_w5\angles_qsd1w5.pkl","rb"))

    sum_ = 0
    num_items = 0
    for ind,item in enumerate(gt):
        for subitem in item:
            sum_ += np.abs(((subitem-angles[ind])+90)%180-90)
            print(ind,subitem,angles[ind],np.abs(((subitem-angles[ind])+90)%180-90))
            num_items += 1

    print("Mean angular error:",sum_*1.0/num_items)


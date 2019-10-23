import os
from glob import glob
import cv2
import numpy as np
import time

        
def TextBoxRemoval(img):
    # Parameters
    resize_size = (1000,1000)
    rectangle_max_min_difference = 10
    th_mean = 63
    min_ratio = 0.08

    # Resize image
    original_size = img.shape[:2]
    img = cv2.resize(img,resize_size)

    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Opening and closing separately
    kernel = np.ones((10,50),np.uint8)
    dark = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('../qsd1_w3/00005_dark.png', cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY))
    cv2.imwrite('../qsd1_w3/00005_bright.png', cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY))

    # Search for largest uniform rectangle on both opening and closing
    rectangles = {"bright": None, "dark": None}
    for mask,mask_name in zip([bright, dark],["bright","dark"]):

        finished = False
        start_row = 0
        mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        while not finished:
            last_row_ind = -1
            last_col_width = -1
            for col_width in range(int(0.5*mask.shape[1]),int(0.1*mask.shape[1]),-2):
                last_col_width = col_width
                for row_ind in range(start_row,mask.shape[0]-int(0.02*mask.shape[0])):
                    indexes = [row_ind,row_ind+int(0.02*mask.shape[0]),int(0.5*mask.shape[1])-int(col_width*0.5),int(0.5*mask.shape[1])+int(col_width*0.5)]
                    rectangle = mask_gray[indexes[0]:indexes[1],indexes[2]:indexes[3]]
                    m = np.mean(rectangle)
                    if mask_name == "bright":
                        cond = m >= th_mean
                    else:
                        cond = m < th_mean
                    if np.max(rectangle)-np.min(rectangle) < rectangle_max_min_difference and cond:
                        last_row_ind = row_ind
                        break
                if last_row_ind != -1:
                    break

            if last_row_ind == -1:
                break

            for col_width in range(last_col_width,mask.shape[1],2):
                last_col_width = col_width
                indexes = [last_row_ind,last_row_ind+int(0.02*mask.shape[0]),int(0.5*mask.shape[1])-int(col_width*0.5),int(0.5*mask.shape[1])+int(col_width*0.5)]
                rectangle = mask_gray[indexes[0]:indexes[1],indexes[2]:indexes[3]]
                m = np.mean(rectangle)
                if mask_name == "bright":
                    cond = m >= th_mean
                else:
                    cond = m < th_mean
                if np.max(rectangle)-np.min(rectangle) > rectangle_max_min_difference and cond:
                    last_col_width = col_width-1
                    break

            for row_length in range(int(0.02*mask.shape[0]),mask.shape[0]-int(0.02*mask.shape[0])):
                indexes = [last_row_ind,last_row_ind+row_length,int(0.5*mask.shape[1])-int(last_col_width*0.5),int(0.5*mask.shape[1])+int(last_col_width*0.5)]
                rectangle = mask_gray[indexes[0]:indexes[1],indexes[2]:indexes[3]]
                m = np.mean(rectangle)
                if mask_name == "bright":
                    cond = m >= th_mean
                else:
                    cond = m < th_mean
                if np.max(rectangle)-np.min(rectangle) > rectangle_max_min_difference and cond:
                    break

            ratio = row_length*original_size[0]/((int(0.5*mask.shape[1])+int(last_col_width*0.5)-int(0.5*mask.shape[1])+int(last_col_width*0.5))*original_size[1])
            if ratio > min_ratio:
                finished = True
                rectangles[mask_name] = [last_row_ind,row_length,last_col_width]
            else:
                start_row = last_row_ind+row_length+1

    # Deciding which rectangle to use (the one from opening or closing)
    if rectangles["bright"] is None and rectangles["dark"] is None:
        mask = np.ones(shape=(mask.shape[0],mask.shape[1]),dtype=np.uint8)*255
        mask = cv2.resize(mask,(original_size[1],original_size[0]))
        return mask, [[0,0],[mask.shape[1],mask.shape[0]]]
    elif rectangles["bright"] is None and rectangles["dark"] is not None:
        print("1")
        last_row_ind = rectangles["dark"][0]
        row_length = rectangles["dark"][1]
        last_col_width = rectangles["dark"][2]
    elif rectangles["bright"] is not None and rectangles["dark"] is None:
        print("2")
        last_row_ind = rectangles["bright"][0]
        row_length = rectangles["bright"][1]
        last_col_width = rectangles["bright"][2]
    else:
        print("3")
        last_row_ind = rectangles["bright"][0]
        row_length = rectangles["bright"][1]
        last_col_width = rectangles["bright"][2]
        indexes = [last_row_ind,last_row_ind+row_length,int(0.5*mask.shape[1])-int(last_col_width*0.5),int(0.5*mask.shape[1])+int(last_col_width*0.5)]
        bright_rect = img[indexes[0]:indexes[1],indexes[2]:indexes[3]]
        mean_0_bright = np.mean(bright_rect[:,:,0])
        mean_1_bright = np.mean(bright_rect[:,:,1])
        mean_2_bright = np.mean(bright_rect[:,:,2])

        last_row_ind = rectangles["dark"][0]
        row_length = rectangles["dark"][1]
        last_col_width = rectangles["dark"][2]
        indexes = [last_row_ind,last_row_ind+row_length,int(0.5*mask.shape[1])-int(last_col_width*0.5),int(0.5*mask.shape[1])+int(last_col_width*0.5)]
        dark_rect = img[indexes[0]:indexes[1],indexes[2]:indexes[3]]
        mean_0_dark = np.mean(dark_rect[:,:,0])
        mean_1_dark = np.mean(dark_rect[:,:,1])
        mean_2_dark = np.mean(dark_rect[:,:,2])

        means_bright = (np.abs(mean_0_bright-mean_1_bright)+np.abs(mean_0_bright-mean_2_bright)+np.abs(mean_2_bright-mean_1_bright))/3
        means_dark = (np.abs(mean_0_dark-mean_1_dark)+np.abs(mean_0_dark-mean_2_dark)+np.abs(mean_2_dark-mean_1_dark))/3

        if means_dark < means_bright:
            last_row_ind = rectangles["dark"][0]
            row_length = rectangles["dark"][1]
            last_col_width = rectangles["dark"][2]
        else:
            last_row_ind = rectangles["bright"][0]
            row_length = rectangles["bright"][1]
            last_col_width = rectangles["bright"][2]

        last_row_ind = rectangles["bright"][0]
        row_length = rectangles["bright"][1]
        last_col_width = rectangles["bright"][2]

    tl = [last_row_ind,int(0.5*mask.shape[1])-int(last_col_width*0.5)]
    br = [last_row_ind+row_length,int(0.5*mask.shape[1])+int(last_col_width*0.5)]
    tl[0] = int(tl[0]*original_size[0]/resize_size[0])
    tl[1] = int(tl[1]*original_size[1]/resize_size[1])
    br[0] = int(br[0]*original_size[0]/resize_size[0])
    br[1] = int(br[1]*original_size[1]/resize_size[1])
    mask = np.ones(shape=(mask.shape[0],mask.shape[1]),dtype=np.uint8)*255
    mask = cv2.resize(mask,(original_size[1],original_size[0]))
    mask[tl[0]:br[0],tl[1]:br[1]] = 0

    return mask, [tl,br]


def main():
    for img_path in glob(os.path.join(r"C:\Users\PC\Documents\Roger\Master\M1\Project\Week2\qsd1_w2","*.jpg"))[:]:
        print("\nUsing img_path",img_path)
        start = time.time()
        img = cv2.imread(img_path)
        mask = TextBoxRemoval(img)
        cv2.imwrite(img_path.replace(".jpg","_mask.png"),mask)
        print("time used:",str(time.time()-start))

if __name__ == '__main__':
    main()

# -- IMPORTS -- #
from copy import copy
import numpy as np
import pickle
import cv2
import time

class TextDetection():
    """CLASS::TextDetection:
        >- Detects the text box in an image."""
    def __init__(self,img_list):
        self.img_list = img_list
        self.text_masks = []

    def detect(self):
        for k,img in enumerate(self.img_list):
            self.text_masks.append([])
            for p,paint in enumerate(img):
                mask = self.detect_text(paint,k,p)
                self.text_masks[-1].append(mask) 
            print('Image ['+str(k)+'] Processed.') 
        return self.text_masks

    def detect_text(self,paint,k,p):
        #paint = cv2.resize(paint,(512,512),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(paint,cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,20))
        white = cv2.morphologyEx(paint,cv2.MORPH_CLOSE,kernel)
        black = cv2.morphologyEx(paint,cv2.MORPH_OPEN,kernel)
        wgray = cv2.cvtColor(white,cv2.COLOR_BGR2GRAY)
        bgray = cv2.bitwise_not(cv2.cvtColor(black,cv2.COLOR_BGR2GRAY))
        _, white_mask = cv2.threshold(wgray, 200, 255, cv2.THRESH_TOZERO)
        _, black_mask = cv2.threshold(bgray, 200, 255, cv2.THRESH_TOZERO)
        white_masked = cv2.bitwise_and(gray, gray, mask=white_mask)
        black_masked = cv2.bitwise_and(inv, inv, mask=black_mask)
        _, white_img = cv2.threshold(white_masked, 200, 255, cv2.THRESH_TOZERO)
        _, black_img = cv2.threshold(black_masked, 200, 255, cv2.THRESH_TOZERO)
        return self.search_rectangles(white_img,black_img,k,p,paint)
    
    def search_rectangles(self,white,black,k,p,paint):
        white_paint = copy(paint)
        black_paint = copy(paint)
        step=10; start = 200; end = 256
        rectangles = []
        min_width = 150; min_height=20; max_height = 120
        detection_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,20))
        if paint.shape[0] > 1000:
            remove_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))
        else:
            remove_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
        for v,value in enumerate(range(start,end,step)):
            white_found = cv2.inRange(white,value,value+step)
            black_found = cv2.inRange(black,value,value+step)
            white_found = cv2.morphologyEx(white_found,cv2.MORPH_CLOSE,detection_kernel)
            black_found = cv2.morphologyEx(black_found,cv2.MORPH_CLOSE,detection_kernel)
            white_found = cv2.morphologyEx(white_found,cv2.MORPH_OPEN,remove_kernel)
            black_found = cv2.morphologyEx(black_found,cv2.MORPH_OPEN,remove_kernel)
            if (np.sum(white_found,axis=None)!=0):
                white_canny = cv2.Canny(white_found,100,100,apertureSize=3)
                _, white_cont, _ = cv2.findContours(white_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(white_cont)==2:
                    for cnt in white_cont:
                        x,y,w,h = cv2.boundingRect(cnt)
                        if w > min_width and h > min_height and h < max_height:
                            cv2.rectangle(white_paint,(x,y),(x+w,y+h),(0,0,255),5)
                            rectangles.append((x,y,w,h))
                #cv2.imwrite('../results/TextBox/{0}_{1}_{2}.png'.format(k,p,'white'),white_paint)
            if (np.sum(black_found,axis=None)!=0):
                black_canny = cv2.Canny(black_found,100,100,apertureSize=3)
                _, black_cont, _ = cv2.findContours(black_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(black_cont)==2:
                    for cnt in black_cont:
                        x,y,w,h = cv2.boundingRect(cnt)
                        if w > min_width and h > min_height and h < max_height:
                            cv2.rectangle(black_paint,(x,y),(x+w,y+h),(0,0,255),5)
                            rectangles.append((x,y,w,h))
                #cv2.imwrite('../results/TextBox/{0}_{1}_{2}.png'.format(k,p,'black'),black_paint)
        mask = np.uint8(np.ones((paint.shape[0],paint.shape[1])))*255
        if rectangles:
            for rect in rectangles:
                x,y,w,h = rect
                mask[y:y+h,x:x+w] = 0
        #cv2.imwrite('../results/TextBox/{0}_{1}_{2}.png'.format(k,p,'final'),mask)
        return mask


if __name__ == "__main__":
    print('-- TESTING TEXTBOXES --')
    start = time.time()
    with open('../results/splitted.pkl','rb') as ff:
        qs_splitted = pickle.load(ff)
    detector = TextDetection(qs_splitted)
    masks = detector.detect()
    print('-- DONE: Time: '+str(time.time()-start))
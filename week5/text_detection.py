# -- IMPORTS -- #
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
        self.text_boxes = []

    def detect(self):
        for k,img in enumerate(self.img_list):
            self.text_boxes.append([])
            self.text_masks.append([])
            for p,paint in enumerate(img):
                mask,textbox = self.detect_text(paint,k,p)
                self.text_masks[-1].append(mask)
                self.text_boxes[-1].append(textbox)   
        return self.text_masks, self.text_boxes

    def detect_text(self,paint,k,p):
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
        return self.search_rectangles(white_img,black_img,k,p)
    
    def search_rectangles(self,white,black,k,p):
        white_found = np.zeros_like(white)
        black_found = np.zeros_like(black)
        # Convert values from 200-255 to 0-255
        for value in range(201,256,1):
            white_found[white==value] = int((value-201)*(255/54))
            black_found[black==value] = int((value-201)*(255/54))
        """NOW WE KNOW THAT THE RECTANGLE HAS AN HOMOGENOUS COLOR, BUT AT DIFFERENT THRESHOLDS.
            CAN USE OTSU'S BINARIZATION OR FIND MAGIC NUMBERS.
            TO FIND THE RECTANGLE: CANNYclear/FINDCONTOURS, BOUNDINGRECT"""
        cv2.imwrite('../results/TextBox/{0:02}_{1}_{2}.png'.format(k,p,'white'),white_found)
        cv2.imwrite('../results/TextBox/{0:02}_{1}_{2}.png'.format(k,p,'black'),black_found)
        return 0,0


if __name__ == "__main__":
    print('-- TESTING TEXTBOXES --')
    start = time.time()
    with open('../results/splitted.pkl','rb') as ff:
        qs_splitted = pickle.load(ff)
    detector = TextDetection(qs_splitted)
    masks,bboxes = detector.detect()
    print('-- DONE: Time: '+str(time.time()-start))
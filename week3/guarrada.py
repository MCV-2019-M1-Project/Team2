import cv2
import numpy as np

img = cv2.imread('../qsd1_w3/00000.jpg')
img = cv2.resize(img,(1000,1000))

img = cv2.medianBlur(img,3)
img = cv2.GaussianBlur(img,(5,5),0)
kernel = np.ones((10, 50), np.uint8)
dark = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
bright = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('../qsd1_w3/00000_dark.png',cv2.cvtColor(dark,cv2.COLOR_BGR2GRAY))
cv2.imwrite('../qsd1_w3/00000_bright.png',cv2.cvtColor(bright,cv2.COLOR_BGR2GRAY))
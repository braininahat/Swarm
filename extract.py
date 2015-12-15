import numpy as np
import cv2
img=cv2.imread("screenie (2).png")
ORANGE_MIN = np.array([0, 60, 230],np.uint8)
ORANGE_MAX = np.array([10, 80, 255],np.uint8)
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
ret,fr=cv2.threshold(frame_threshed,200,255,0)
height, width = fr.shape
res = cv2.resize(fr,(150,75), interpolation = cv2.INTER_CUBIC)
cv2.imwrite('output233.jpg', res)

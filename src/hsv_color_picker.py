import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("data_test/hill_shadow.jpg")
HSL = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
HSL = cv2.resize(HSL, (640,320))

def getpos(event, x,y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSL: ",HSL[y,x])
        H ,S ,L = HSL[y,x]
        print("S-L Diff: ", S - L)
        h_channel = H
        s_channel = S
        l_channel = L
        # Conditions
        # condition_1 = (100 < h_channel < 120) & (55 < s_channel < 80) & (25 < l_channel < 45)  Dark
        condition_1 = (100 < h_channel) & (h_channel < 120) & \
                      ((s_channel - l_channel) > 20) & \
                      ((s_channel - l_channel) < 40) & \
                      (s_channel > 60)
        # (55<s_channel)&(s_channel<80) & (25<l_channel)&(l_channel<55)
        # condition_2 = (20 < h_channel < 40) & (s_channel > 200) | (l_channel > 200)   Bright
        condition_2 = (15 < h_channel) & (h_channel < 40) & \
                      ((((s_channel - l_channel) > 175) & ((s_channel - l_channel) < 220)) |
                      ((s_channel > 200) & (l_channel > 100)))
        print(condition_1, condition_2)


cv2.imshow("HSV image", HSL)
# cv2.imshow('original', image)
cv2.setMouseCallback("HSV image", getpos)
cv2.waitKey(0)

import cv2
import matplotlib as plt
import numpy as np
import os
import sys
import resizeimage


LeftImgPath = sys.argv[1]
RightImgPath = sys.argv[2]

def checkImage(img):
    if img is None:
        print("image dosent found!")
        exit()
    
RightImg = cv2.imread(RightImgPath)
LeftImg = cv2.imread(LeftImgPath)


checkImage(RightImg)
checkImage(LeftImg)


min_height = min([LeftImg.shape[0], RightImg.shape[0]])
min_width = min([LeftImg.shape[1], RightImg.shape[1]])

dim = (min_width,min_height)

resizedRight = cv2.resize(RightImg, dim, interpolation = cv2.INTER_AREA)
resizedLeft = cv2.resize(LeftImg, dim, interpolation = cv2.INTER_AREA)

GrayRight = cv2.cvtColor(RightImg, cv2.COLOR_BGR2GRAY)
GrayLeft = cv2.cvtColor(LeftImg, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
Kp1 , desc1 = orb.detectAndCompute(GrayRight,None)
Kp2 , desc2 = orb.detectAndCompute(GrayLeft,None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1,desc2)

# print(matches)
goodMatch = []
for m in matches:
    if m.distance < 0.8:
        goodMatch.append(m)
# print(goodMatch)

imageMatches = cv2.drawMatches(Kp1,LeftImg RightImg,Kp2,matches,None)

# im1= cv2.resize(resizedRight,(960,540))
# im2 = cv2.resize(resizedLeft,(960,540))


im1= cv2.resize(imageMatches,(960,540))


cv2.imshow('right',im1)
# cv2.imshow('left',im2)
cv2.waitKey(0)




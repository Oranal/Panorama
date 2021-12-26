import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


LeftImgPath = sys.argv[1]
RightImgPath = sys.argv[2]
outPath = sys.argv[3]

def checkImage(img):
    if img is None:
        print("image dosen't found!")
        exit()
    
RightImg = cv2.imread(RightImgPath)
LeftImg = cv2.imread(LeftImgPath)


checkImage(RightImg)
checkImage(LeftImg)


#calculate the scale between the two images height
min_height = min([LeftImg.shape[0], RightImg.shape[0]])
max_height = max([LeftImg.shape[0], RightImg.shape[0]])
imScale = max_height/min_height


imArr =[]
if min_height == RightImg.shape[0]:
    imArr.append(LeftImg)
    imArr.append(RightImg)
    
    dim = (
    round(imArr[0].shape[1]/imScale),
    round(imArr[0].shape[0]/imScale),
    )
    
    LeftImg = cv2.resize(LeftImg, dim, interpolation = cv2.INTER_AREA)

else:
    imArr.append(RightImg)
    imArr.append(LeftImg)
    
    dim = (
    round(imArr[0].shape[1]/imScale),
    round(imArr[0].shape[0]/imScale),
    )
    
    RightImg = cv2.resize(RightImg, dim, interpolation = cv2.INTER_AREA)


#converting to gray from bgr to detect key points and descriptors
GrayRight = cv2.cvtColor(RightImg, cv2.COLOR_BGR2GRAY)
GrayLeft = cv2.cvtColor(LeftImg, cv2.COLOR_BGR2GRAY)

#  find the keypoints and descriptors with orb.
orb = cv2.ORB_create()
KpLeft , descLeft = orb.detectAndCompute(GrayLeft,None)
KpRight , descRight = orb.detectAndCompute(GrayRight,None)

#matches between the descriptors according to the distance NORM HAMMIN function for orb objects.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descLeft,descRight)


#sorted the 20% of the first matches with the smallest distance.
goodMatch = sorted(matches, key=lambda x: x.distance, reverse=False)[:len(matches)//5]


#draw the lines between key points according to the the matches we found for checking the matches.
imageMatches = cv2.drawMatches(LeftImg,KpLeft,RightImg,KpRight,goodMatch,None)


#calculate the homography matrix according the key points we found and sorted.
src_pts = np.float32([ KpLeft[m.queryIdx].pt for m in goodMatch ]).reshape(-1,1,2)
dst_pts = np.float32([ KpRight[m.trainIdx].pt for m in goodMatch ]).reshape(-1,1,2)
H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

# warp the right image according to the homography matrix and the width of the to images.
result = cv2.warpPerspective(RightImg,H,((RightImg.shape[1] + LeftImg.shape[1]), LeftImg.shape[0]))

# insert left image
result[0:LeftImg.shape[0], 0:LeftImg.shape[1]] = LeftImg
#convert the reslut color to gray for finding the black area contur and remove him
ResGray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# threshold for black pixels
_,thresh = cv2.threshold(ResGray, 1, 255, cv2.THRESH_BINARY)
# find non black pixels region
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

# create rectangle of non black pixels
x,y,w,h = cv2.boundingRect(cnt)

# crop rectangle
result = result[y:y+h,x:x+w]

#saving the panorama result in the output directory path/
cv2.imwrite(os.path.join(outPath ,'Panorama.jpg'),result)



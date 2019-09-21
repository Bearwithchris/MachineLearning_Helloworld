#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:35:42 2019
Resource: https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/?completed=/loading-video-python-opencv-tutorial/
@author: christeo
"""

import numpy as np
import cv2

img=cv2.imread("watch.jpg",cv2.IMREAD_COLOR) #Load image 

##Drawing a line
#cv2.line(img,(0,0),(150,150),(255,255,255),15) #cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

##Drawing a rectangle
#cv2.rectangle(img,(15,25),(200,150),(0,0,255),15) #cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)

##Drawing a cricle
#cv2.circle(img,(100,63), 55, (0,255,0), -1) #cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)


##Drawing octagon/pentagon etc
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32) #Bulild the Array list for the shape
cv2.polylines(img, [pts], True, (0,255,255), 3) #cv2.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]

#Words
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(0,130), font, 1, (200,255,155), 2, cv2.LINE_AA)


cv2.imshow("imgage_watch",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:20:05 2019
Reference: https://pythonprogramming.net/loading-video-python-opencv-tutorial/?completed=/loading-images-python-opencv-tutorial/
@author: christeo
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0) #Loading into webcam zero , alternative can add in a video file

#############Write out the video file#################
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
 
while(True):
    ret, frame = cap.read() #Returns back True/False for feed & frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convertcolor to Gray
 
    ####Output the frame###
#    out.write(frame)
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'): #ord returns the unicode for the char
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
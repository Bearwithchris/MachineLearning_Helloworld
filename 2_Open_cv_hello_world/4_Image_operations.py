#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:31:53 2019
https://pythonprogramming.net/image-operations-python-opencv-tutorial/
@author: christeo
"""

import numpy as np
import cv2

img=cv2.imread("watch.jpg",cv2.IMREAD_COLOR)

#Referecing pixels 
px=img[55:150,55:150]
print (px)
cv2.imshow("image",px)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

########CV2 Commands #########
img=cv2.imread("watch.jpg" ,cv2.IMREAD_GRAYSCALE) #Loading the image in Grayscale i.e. reducing the colour channels 
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('watchgray.png',img)

#Pyplot method###########
#img=cv2.imread("watch.jpg" ,0)
#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
#plt.show()

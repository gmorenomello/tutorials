# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import matplotlib.pylab as plt

# Load an color image in grayscale
fname = r'J:\MEA images Vibeke Ola 2018\2017.03.23 Motoneuron Culture 3D-Gel.tif'
img = cv2.imread(fname,0)

plt.imshow(img,cmap='hot')

fname_ext = fname[-4:]
dest_fname = fname[:-4] + '_mask' +fname_ext

ret,thresh = cv2.threshold(img,28,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 5)
dilation = cv2.dilate(thresh,kernel,iterations = 12)
plt.imshow(thresh)
plt.show()
cv2.imwrite(dest_fname,thresh)

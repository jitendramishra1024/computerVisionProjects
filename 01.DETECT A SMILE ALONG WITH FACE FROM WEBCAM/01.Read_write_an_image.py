# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:57:18 2020

@author: INE12363221
"""

import cv2 

input = cv2.imread('./image_examples/elephant.jpg')

cv2.imshow('Test Elephant Image', input)
cv2.waitKey()
cv2.destroyAllWindows()

# press any key on pop up window to close it 

# Let's print each dimension of the image

print('Height of Image:', int(input.shape[0]), 'pixels')
print('Width of Image: ', int(input.shape[1]), 'pixels')

# make the image gray and show 
gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
cv2.imshow('Test Elephant Image', gray)
cv2.waitKey()
cv2.destroyAllWindows()

#save gray color to  a different file
cv2.imwrite('./image_examples/gray_elephant.jpg', gray)
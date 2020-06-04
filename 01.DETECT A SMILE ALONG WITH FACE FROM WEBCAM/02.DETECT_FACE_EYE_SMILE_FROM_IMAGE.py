# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:00:51 2020

@author: INE12363221
"""
#REF:
#PART1: https://towardsdatascience.com/the-intuition-behind-facial-detection-the-viola-jones-algorithm-29d9106b6999
#PART2: https://towardsdatascience.com/facial-recognition-happiness-bbb3c4293d1d
import numpy as np
import cv2

# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml') # We load the cascade for the eyes.
smile_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_smile.xml') # We load the cascade for the eyes.

# Load our image then convert it to grayscale
image = cv2.imread('image_examples/obama.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20) 
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            cv2.rectangle(roi_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (200, 0, 130), 2)
    return frame # We return the image with the detector rectangles.

img=detect(gray,image)
cv2.imshow('img',img)
cv2.waitKey(0)
    
cv2.destroyAllWindows()
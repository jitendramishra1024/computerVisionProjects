# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:13:00 2020

@author: INE12363221
"""

from tensorflow.python.keras.models import load_model 
import cv2
import numpy as np
import tensorflow as tf

model = load_model("rock-paper-scissors-model.h5")
cap = cv2.VideoCapture(0)

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]
print(tf.__version__)

while True:
    ret, frame = cap.read()
    #if not ret:
        #continue
    print("inside while loop")
    # rectangle for user to play
    cv2.rectangle(frame, (70, 70), (300, 300), (255, 255, 255), 2)
    # extract the region of image within the user rectangle
    roi = frame[70:300, 70:300]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 200))
    img = tf.cast(img, tf.float32)
   # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    
        # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    k = cv2.waitKey(10)
    cv2.imshow("Rock Paper Scissors", frame)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
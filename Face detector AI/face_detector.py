import cv2 as cv
import numpy as np
from random import randrange

#Load some pre-trained data on face frontals from open cv
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img = cv.imread('morgan.jpg')

webcam = cv.VideoCapture(0)
print(webcam.isOpened()) # False
print(webcam.read()) # (False, None)

while True:

  successful_frame_read, frame = webcam.read()

  #Convert to grayscale
  grayscale_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  #detect faces
  face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

  for (x,y,w,h) in face_coordinates:
    cv.rectangle(frame, (x, y),(x+w,y+h), (0,255,0), 2)
  
  cv.imshow('Face detector',frame)
  key = cv.waitKey(1)
  
  if key == 81 or key == 113:
    break

webcam.release()




print("code completed")

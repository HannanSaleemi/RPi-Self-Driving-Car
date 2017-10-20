import numpy as np
import cv2

stopCascade = cv2.CascadeClassifier('stop_class.xml')

img = cv2.imread('stopimage.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stop = stopCascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in stop:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]

cv2.imshow('img',roi_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()







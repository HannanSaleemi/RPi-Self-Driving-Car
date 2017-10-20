import numpy as np
import cv2

#Haar Cascade Detection
stopCascade = cv2.CascadeClassifier('traffic.xml')

#Red or Green Images
img = cv2.imread('greentraffic.jpg')
#img = cv2.imread('redtraffic.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stop = stopCascade.detectMultiScale(gray, 1.1, 2)

for (x,y,w,h) in stop:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]

#cv2.imshow('img',roi_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Try to find the brightest spot
#Apply guassian blur to ROI in order to remove highe frequency noise from the image
newgray = cv2.GaussianBlur(roi_gray, (41, 41), 0)

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(newgray)

cv2.circle(roi_gray, maxLoc, 42, (255, 0, 0), 2)

print(maxLoc)

cv2.imshow("IMG", roi_gray)
cv2.waitKey(0)

#RED maxLoc = (149,148)
#RED seems to be in the y range of (100-170)
#GREEN maxLoc = (115, 272)
#GREEN seems to be in the y range of (230-295)
#Add an undiecided factor if it falls outside of these ranges










import cv2
import DataPreProcessing
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage.transform import resize

roi_gray = 0
stopPresent = False

#STOP Sign Detection
def stopDetection(img):
    global roi_gray
    global stopPresent
    stopCascade = cv2.CascadeClassifier('stop_class.xml')
    stop = stopCascade.detectMultiScale(img,
                                        scaleFactor=1.1,
                                        minNeighbors=5
                                        )
    for (x, y, w, h) in stop:
        stopPresent = True
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    print("STOP Detection Complete")
    print(stopPresent)


#recv_img = misc.imread('/Volumes/TRANSCEND/RPi-Self-Driving-Car/Testing/STOP-testing/image.png')

#img = np.array(recv_img, dtype='uint8')
img = cv2.imread('img.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#image_resized = resize(gray, (18,22), mode='reflect')
stopDetection(gray)
plt.imshow(gray, cmap='Greys')
plt.show()

'''
import numpy as np
import cv2

stopCascade = cv2.CascadeClassifier('stop_class.xml')

img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stop = stopCascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in stop:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]

cv2.imshow('img',roi_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

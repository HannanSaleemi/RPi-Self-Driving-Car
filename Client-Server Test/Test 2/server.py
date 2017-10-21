import socket               
import cv2
import numpy as np
import threading

stopPresent = False
greenPresent = False
redPresent = False

def stopDetection():
    #Make the image public so that before these threaded tasks run the img is open to avoid having 3 of the same image being opened
    stopCascade = cv2.CascadeClassifier('stop_class.xml')

    img = cv2.imread('torecv.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop = stopCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in stop:
##        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        stopPresent = True

def trafficLightDetection():
    trafficCascade = cv2.CascadeClassifier('traffic.xml')
    img = cv2.imread('torecv.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    traffic = trafficCascade.detectMultiScale(gray, 1.1, 2)

    for (x,y,w,h) in traffic:
        print("Detected Traffic")
        #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]

        newgray = cv2.GaussianBlur(roi_gray, (41, 41), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(newgray)

        print(maxLoc)
        print(maxLoc[0])
        print(maxLoc[1])

        if maxLoc[1] >= 230:
            print("Green Detected")
            greenPresent = True
        elif maxLoc[1] >= 100:
            print("Red Detected")
            redPresent = True



s = socket.socket()         
host = 'localhost' 
port = 12347                 
s.bind(('192.168.0.60', port))        
f = open('torecv.png','wb')
s.listen(1)                 
try:
    while True:
        c, addr = s.accept()     
        print ('Got connection from', addr)
        print ("Receiving...")
        l = c.recv(150000)
        while (l):
            print ("Receiving...")
            f.write(l)
            l = c.recv(150000)
        f.close()
        print ("Done Receiving")
        c.send(b'Thank you for connecting')
        c.close()
        break

    #Neural Network Predition START
            
    
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")


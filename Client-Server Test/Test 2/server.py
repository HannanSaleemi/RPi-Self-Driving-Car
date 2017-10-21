import socket               
import cv2
import numpy as np
import threading

stopPresent = False
greenPresent = False
redPresent = False

stopComplete = False
trafficComplete = False

def stopDetection():
    #Make the image public so that before these threaded tasks run the img is open to avoid having 3 of the same image being opened
    stopCascade = cv2.CascadeClassifier('stop_class.xml')
    stop = stopCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in stop:
        roi_gray = gray[y:y+h, x:x+w]
        stopPresent = True

    stopComplete = True
    print("STOP complete")

def trafficLightDetection():
    trafficCascade = cv2.CascadeClassifier('traffic.xml')
    traffic = trafficCascade.detectMultiScale(gray, 1.1, 2)

    for (x,y,w,h) in traffic:
        print("Detected Traffic")
        roi_gray = gray[y:y+h, x:x+w]

        #Detection
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
    trafficComplete = True
    print("Traffic Complete")
    



s = socket.socket()         
host = '192.168.0.60' 
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

    #Open and convert image to grey for threaded processes to use
    img = cv2.imread('torecv.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stopThread = threading.Thread(target=stopDetection)
    trafficThread = threading.Thread(target=trafficLightDetection)

    stopThread.start()
    trafficThread.start()

    stopThread.join()
    trafficThread.join()

    print("ALL DONE!!")

    
    #Neural Network Predition START
            
    
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")


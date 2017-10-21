import socket               
import cv2
import numpy as np

stopPresent = False

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
    
    #Stop Sign
    stopCascade = cv2.CascadeClassifier('stop_class.xml')

    img = cv2.imread('torecv.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop = stopCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in stop:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        stopPresent = True
        
    print("STOP Sign Present", stopPresent)
    
    #Traffic Light Threaded
    
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")


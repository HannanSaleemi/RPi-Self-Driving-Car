import socket               
import cv
import numpy as np

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
    
    #Traffic Light Threaded
    
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")


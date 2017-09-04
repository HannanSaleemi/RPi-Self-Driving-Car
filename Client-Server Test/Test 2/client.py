import cv2
import datetime
import socket
from time import sleep

s = socket.socket()
s.connect(('192.168.0.10', 12346))

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,200)

img_counter = 0

for x in range(0,10):
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    now = datetime.datetime.now()
    img_name = "frame {}.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    f = open(img_name, "rb")
    l = f.read(90024)
    while (l):
        print ("Sending...")
        s.send(l)
        l = f.read(90024)
    f.close()
    print ("IMG SENT...")
    img_counter += 1

print ("Exiting...")
cap.release()
cv2.destroyAllWindows()
s.shutdown(socket.SHUT_WR)

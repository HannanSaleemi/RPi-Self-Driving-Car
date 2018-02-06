from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import cv2
from socket import *
import numpy as np
from skimage.transform import resize
from scipy import misc
import matplotlib.pyplot as plt

#TCP Send Subroutinues
def send_img(dest):
    f = open('img.png', 'rb')
    print("[*] Sending Image...")
    l = f.read(150000)
    while (l):
        dest.send(l)
        l = f.read(150000)
    print("[*] Image Sucessfully Sent")

#TCP Recvieve Subroutinue
def recv_results(conn):
    data = conn.recv(1024)
    print(data)
    
# Camera setup
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = (32)
rawCapture = PiRGBArray(camera, size=(320, 240))

#Results
results = ""

#TCP Setup
print("[*] Waiting for connection...")
conn = socket(AF_INET, SOCK_STREAM)
conn.connect(('192.168.0.60', 25000))
#conn.connect(('10.124.128.144', 25000))
print("[*] Successfully connected")

# Camera Warmup
sleep(1)

# Main stream capture loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # Image is stored in an array and saved to file
    image = frame.array
    cv2.imwrite('img.png', image)

    #Send the image
    send_img(conn)

    sleep(5)

    #Wait for results to come back from server
    recv_results(conn)
    
    # Reset the capture array ready for next frame
    rawCapture.truncate(0)

    break

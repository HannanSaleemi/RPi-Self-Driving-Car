from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import cv2
from socket import *
import numpy as np
from skimage.transform import resize
from scipy import misc
import matplotlib.pyplot as plt
import serial

#TCP Send Subroutinues
def send_img(dest):
    f = open('img.png', 'rb')
    #print("[*] Sending Image...")
    l = f.read(150000)
    while (l):
        dest.send(l)
        l = f.read(150000)
    #print("[*] Image Sucessfully Sent")

#TCP Recvieve Subroutinue
def recv_results(conn):
    data = conn.recv(4096)
    data = data.decode('utf-8')
    print(data)
    return data

#TCP New connection
def init_new_conn():
    conn = socket(AF_INET, SOCK_STREAM)
    #print("[*] Waiting for connection...")
    #conn.connect(('10.124.168.149', 25000))
    conn.connect(('192.168.0.60', 25000))
    #print("[*] Successfully Connected!")
    return conn

# Camera setup
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = (32)
rawCapture = PiRGBArray(camera, size=(320, 240))
#Results
results = ""

#Connect to Arduino
print("[*] Connecting to Arduino...")
arduino = serial.Serial('/dev/ttyACM0', 115200)
print("[*] Successfully Connected to Arduino")
arduino.flushInput()

# Camera Warmup
sleep(1)

# Main stream capture loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #Connect to server
    conn = init_new_conn()
    
    # Image is stored in an array and saved to file
    image = frame.array
    cv2.imwrite('img.png', image)

    #Send the image
    send_img(conn)
    conn.close()

    sleep(0.5)

    #Start the new connection
    conn = init_new_conn()
    #Wait for results to come back from server
    results = recv_results(conn)
    #Close The Connection
    conn.close()
    
    # Reset the capture array ready for next frame
    rawCapture.truncate(0)

    #Send the result to Arduino
    results = bytes(results.encode('ascii'))
    arduino.write(results)

    

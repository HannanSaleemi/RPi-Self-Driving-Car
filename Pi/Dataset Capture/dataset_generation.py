from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = (32)
rawCapture = PiRGBArray(camera, size=(320, 240))

time.sleep(0.1)
print("[*] STARTING IN 5...")
time.sleep(5)
img_counter = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Frame", image)
    img_name = "forward {}.png".format(img_counter)
    cv2.imwrite(img_name, image)
    print("DONE " + img_name)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    img_counter += 1
    if key == ord("q"):
        break

import cv2
from time import sleep

cam = cv2.VideoCapture(0)
cam.set(3,500)
cam.set(4,200)

cv2.namedWindow("test")

img_counter = 0

sleep(1)
print("-----PLACE CAR ON TRACK-----")
sleep(1)
print("-----PLACE CAR ON TRACK - 1 seconds left...-----")
sleep(1)
print("Staring...")

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape Hit, closing...")
        break
    else:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()


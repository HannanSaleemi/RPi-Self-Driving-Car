import cv2
import sys

#Importing the trained cascade file
cascFile = "stop_class.xml"
#Setting the cascade file
stopCascade = cv2.CascadeClassifier(cascFile)

#Set the device to capture video frames from
#In this case, the webcam
cap = cv2.VideoCapture(0)

#UNSUCCESSFULLY, setting the hight and the width of the webcam
cap.set(3, 600)
cap.set(4, 800)


while True:
    #Start to capture frames from webcam
    ret, frame = cap.read()

    #Convert each frame into greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the faces using the cascade
    #First parameter -> the greyscale image
    #Second parameter -> the scale factor - change to avoid detecting the wrong things
    faces = stopCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    #For each face that it found
    for (x, y, w, h) in faces:
        #Create a green rectange around the face, with a border width of 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Display the video stream with the squares draw on
    cv2.imshow('Video', frame)

    #If at anypoint the letter q is pressed, quit the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Cleanup the session
cap.release()
cv2.destroyAllWindows()

import cv2
import sys
# TODO - change var names as taken from somewhere
#Import the cascade file
cascPath = "stop_class.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#Start video capture
video_capture = cv2.VideoCapture(0)

#UNSUCCESSFULLY set the video hieght and width
video_capture.set(3,800)
video_capture.set(4,600)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #Convert to GreyScale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Start to find the stop signs
    #Tune the parameters in here to make the classifier more accurate
    #Scale factor will fix any issues where the classifier is puting the box around the wrong things
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Wait untill the q key is pressed in order to break the while loop and
    # safely exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

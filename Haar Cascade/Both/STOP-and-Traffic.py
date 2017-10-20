import cv2
import sys

#Importing the trained cascade file
'''cascFile = "traffic.xml"'''
trafficCascFile = "traffic.xml"
stopCascFile = "stop_class.xml"
#Setting the cascade file
stopCascade = cv2.CascadeClassifier(stopCascFile)
trafficCascade = cv2.CascadeClassifier(trafficCascFile)

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
    stop = stopCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    traffic = trafficCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40,90)
    )

    #For each stop sign that it found
    for (x, y, w, h) in stop:
        #Create a green rectange around the face, with a border width of 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        #Display text to show shtop sign
        '''cv2.putText(frame, "GREEN", (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)'''

    #For each traffic light is found
    for (x, y, w, h) in traffic:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    #Display the video stream with the squares draw on
    cv2.imshow('Video', frame)

    #If at anypoint the letter q is pressed, quit the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Cleanup the session
cap.release()
cv2.destroyAllWindows()

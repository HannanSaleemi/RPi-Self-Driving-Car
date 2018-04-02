from socket import *
import numpy as np
import matplotlib.pyplot as plt
import DataPreProcessing
import threading
import tensorflow as tf
from scipy import misc
import cv2
from time import sleep
import time

#Result variables initalisation
lightColor = ""
stopPresent = "F"
pred_result = [0]
results_array = np.array(["N", "N", "N"])

#Stop Sign Detection Variables and constants
stop_distance = 100
stop_start = 0
stop_finish = 0
stop_time = 0
drive_time_after_stop = 0
stop_sign_active = True
stop_flag = False
focal = 318.88
stop_actualWidth = 4.5

#Traffic Lights variables
traffic_actualWidth = 3.9
traffic_distance = 100

#TCP Recieve method
def recv_img(conn):
    f = open('img.png', 'wb')
    print("[*] Recieving...")
    l = conn.recv(150000)
    while (l):
        f.write(l)
        l = conn.recv(150000)
    print("[*] Image Successfully Recieved")

#Initalise a new connection to the server
def init_new_conn():
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('192.168.0.60', 25000))
    #s.bind(('10.124.168.149', 25000))
    print("[*] Waiting for connection...")
    s.listen(5)
    return s

#Establish a link with client
def listen_new_conn(s):
    conn, a = s.accept()
    print("[*] Connected to Client!")
    return conn

#TCP Send Results
def send_results(directionResult, trafficResult, stopResult, dest):
    completeResult = str(directionResult) + str(trafficResult) + str(stopResult)
    print("[*] Sending", completeResult)
    completeResult = str.encode(completeResult)
    dest.send(completeResult)
    print("[*] Result sent")

#Neural Network Function
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#Directional prediction method
def directionPrediction(image):
    global pred_result
    global multilayer_perceptron
    n_input = 396
    n_classes = 3
    n_hidden_1 = 256
    n_hidden_2 = 256
    weights = {
        'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_classes]))
    }
    x = tf.placeholder('float', [None, n_input])
    pred = multilayer_perceptron(x, weights, biases)

    recv_image = np.zeros(shape=396)
    recv_image = image.reshape(1, 396)
    init = tf.global_variables_initializer()
    #Predict
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")
        print("[*] Model Restored")
        result = sess.run(tf.argmax(pred, 1), feed_dict={x: recv_image})
        print("[*] Model prediction result: ", str(result))
        pred_result = result

#STOP Sign Detection
def stopDetection():
    global stopPresent, stop_sign_active, stop_flag
    global stop_start, stop_finish, stop_time, drive_time_after_stop
    stopCascade = cv2.CascadeClassifier('stop_class.xml')
    stop = stopCascade.detectMultiScale(grey,
                                        scaleFactor=1.1,
                                        minNeighbors=2
                                        )
    for (x, y, w, h) in stop:
        #Calculate Distance
        stop_distance = (stop_actualWidth * focal) / w
        print("Distance:", stop_distance)

        #If distance is less than 25cm
        if 0 < stop_distance < 30 and stop_sign_active:
            print("STOP SIGN AHEAD")
            stopPresent = "T"

            #Storing the first time to stop the vehicle
            if stop_flag is False:
                stop_start = cv2.getTickCount()
                stop_flag = True

            #Each loop a new time is recorded to compare against start time
            stop_finish = cv2.getTickCount()
            stop_time = (stop_finish - stop_start) / cv2.getTickFrequency()
            print("STOP Halted for:", stop_time)

            #If 5 seconds are up, begin to drive and set ignore flags
            if stop_time > 5:
                print("5 Seconds up, continue driving...")
                stop_flag = False
                stop_sign_active = False

        else:
            #Continue with driving
            stopPresent = "F"
            stop_start = cv2.getTickCount()
            stop_distance = 100
            if stop_sign_active is False:
                drive_time_after_stop = (stop_start - stop_finish) / cv2.getTickFrequency()
                if drive_time_after_stop > 15:
                    stop_sign_active = True

    print("[*] STOP Detection Complete")
    print("[*] STOP sign present:", stopPresent)

def trafficLightDetection():
    global lightColor
    trafficCascade = cv2.CascadeClassifier('traffic.xml')
    traffic = trafficCascade.detectMultiScale(grey, 1.1, 2)

    for (x,y,w,h) in traffic:
        print("[*] Detected Traffic Light")
        roi_gray = grey[y:y+h-20, x:x+w]

        #Blur to remove any noise
        blurred = cv2.GaussianBlur(roi_gray, (41, 41), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)

        #Coordinates of the brightest light
        print(maxLoc[1])

        #Get Distance
        traffic_distance = (traffic_actualWidth * focal) / w
        print("Distance:", traffic_distance)

        #If close enough - Identify the light color
        if 0 < traffic_distance < 40:
            if maxLoc[1] >= 40:#used to be 35
                print("[*] Green Detected")
                lightColor = "G"
            elif maxLoc[1] <= 40:#used to br 30
                print("[*] Red Detected")
                lightColor = "R"
        else:
            print("[*] Traffic Light Not Close Enough")
            lightColor = "N"

        print("[*] Traffic Light Detected")
    print("[*] Traffic Light Detection Complete")


#Initalising the TCP Connection
s = init_new_conn()

try:
    while True:
        #Accept connections from client
        conn = listen_new_conn(s)

        #Variable reset
        stopPresent = "F"
        lightColor = "N"
        pred_result = [0]

        #Creating the recieving array
        recieved_array = np.zeros((240, 320, 3))
        recv_img(conn)
        conn.close()
        start = time.time()
        img = cv2.imread('img.png')

        #Image downsize Greyscale conversion - directional
        img_converter = DataPreProcessing.DatasetProcessing()
        downsized_img = img_converter.getSingleImage(img)

        #Image greyscale only - stop sign and traffic light
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Initalise threads
        directionThread = threading.Thread(target=directionPrediction, args=(downsized_img))
        stopThread = threading.Thread(target=stopDetection)
        trafficThread = threading.Thread(target=trafficLightDetection)

        #Start the threads
        directionThread.start()
        stopThread.start()
        trafficThread.start()

        #Join the queue - so they all wait to finish before sending the result
        directionThread.join()
        stopThread.join()
        trafficThread.join()

        end = time.time()

        #Results sending
        conn = listen_new_conn(s)
        send_results(pred_result[0], lightColor, stopPresent, conn)
        conn.close()

        print(end - start)

except KeyboardInterrupt:
    print("[*] Connection being closed...")
finally:
    conn.close()

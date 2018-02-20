from socket import *
import numpy as np
import matplotlib.pyplot as plt
import DataPreProcessing
import threading
import tensorflow as tf
from scipy import misc
import cv2
from time import sleep

#Resul variables initalisation
lightColor = ""
stopPresent = "F"
pred_result = [0]
results_array = np.array(["N", "N", "N"])

#TCP Recieve method
def recv_img(conn):
    f = open('img.png', 'wb')
    print("[*] Recieving...")
    l = conn.recv(150000)
    while (l):
        f.write(l)
        l = conn.recv(150000)
    print("[*] Image Successfully Recieved")


#s.bind(('localhost', 25000))
#s.bind(('10.124.136.32', 25000))

#Initalise a new connection to the server
def init_new_conn():
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('192.168.0.60', 25000))
    #s.bind(('10.124.144.232', 25000))
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
    global stopPresent
    stopCascade = cv2.CascadeClassifier('stop_class.xml')
    stop = stopCascade.detectMultiScale(grey,
                                        scaleFactor=1.1,
                                        minNeighbors=2
                                        )
    for (x, y, w, h) in stop:
        stopPresent = "T"
    print("[*] STOP Detection Complete")
    print("[*] STOP sign present:", stopPresent)

def trafficLightDetection():
    global lightColor
    trafficCascade = cv2.CascadeClassifier('traffic.xml')
    traffic = trafficCascade.detectMultiScale(grey, 1.1, 2)

    for (x,y,w,h) in traffic:
        print("[*] Detected Traffic Light")
        roi_gray = grey[y:y+h, x:x+w]

        blurred = cv2.GaussianBlur(roi_gray, (41, 41), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)

        print(maxLoc)
        print(maxLoc[0])
        print(maxLoc[1])

        #Not working well - replace with y + (height / 2) then the usual greater and less than

        if maxLoc[1] >= 35:
            print("[*] Green Detected")
            lightColor = "G"
        elif maxLoc[1] <= 30:
            print("[*] Red Detected")
            lightColor = "R"
        else:
            lightColor = "N"
        print("[*] Traffic Light Detected")
    print("[*] Traffic Light Detection Complete")


#Initalising the TCP Connection
s = init_new_conn()

import time

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

        break

except KeyboardInterrupt:
    print("[*] Connection being closed...")
finally:
    conn.close()

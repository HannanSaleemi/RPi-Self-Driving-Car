#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import socket               
import cv2
import numpy as np
import threading
import tensorflow as tf
from scipy import misc
from skimage.transform import resize

stopPresent = False
greenPresent = False
redPresent = False
pred_result = [5]

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def nnPrediction(image):
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
    
    array_test = np.zeros(shape=396)
    array_test = image.reshape(1,396)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")
        print("Model Restored")
        result = sess.run(tf.argmax(pred, 1), feed_dict={x: array_test})
        print(result)
        pred_result = result

    

def stopDetection():
    #Make the image public so that before these threaded tasks run the img is open to avoid having 3 of the same image being opened
    global stopPresent
    stopCascade = cv2.CascadeClassifier('stop_class.xml')
    stop = stopCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30,30)
                                        )

    for (x,y,w,h) in stop:
        #roi_gray = gray[y:y+h, x:x+w]
        stopPresent = True

    print("STOP complete")

def trafficLightDetection():
    global redPresent
    global greenPresent
    trafficCascade = cv2.CascadeClassifier('traffic.xml')
    traffic = trafficCascade.detectMultiScale(gray, 1.3, 2)
    #traffic = trafficCascade.detectMultiScale(gray,
    #                                          scaleFactor=1.1,
    #                                          minNeighbors=5,
    #                                          minSize=(30,30)
    #                                          )

    for (x,y,w,h) in traffic:
        print("Detected Traffic")
        roi_gray = gray[y:y+h, x:x+w]

        #Detection
        newgray = cv2.GaussianBlur(roi_gray, (41, 41), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(newgray)

        print(maxLoc)
        print(maxLoc[0])
        print(maxLoc[1])

        #Get height - divide by 2 then top half is red and bottom half is green

        if maxLoc[1] >= 210:
            print("Green Detected")
            greenPresent = True
        elif maxLoc[1] <= 160:
            print("Red Detected")
            redPresent = True
    print("Traffic Complete")
    




s = socket.socket()         
host = '192.168.0.60' 
port = 12347                 
s.bind(('192.168.0.60', port))        
f = open('torecv.png','wb')
s.listen(1)                 
try:
    while True:
        c, addr = s.accept()     
        print ('Got connection from', addr)
        print ("Receiving...")
        l = c.recv(150000)
        while (l):
            print ("Receiving...")
            f.write(l)
            l = c.recv(150000)
        f.close()
        print ("Done Receiving")
        #c.send(b'Thank you for connecting')
        #c.close()
        break

    #IMPORT TENSORFLOW and load the model now
    #Maybe do this when initalising the server at the start
    #Then THREAD the predicting process and join
    #Combine all results and send across to the

    #Open and convert image to grey for threaded processes to use
    img = cv2.imread('torecv.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    image = misc.imread('/Volumes/TRANSCEND/RPi-Self-Driving-Car/Models/Jupyter/v0.4 - 1st Iteration/torecv.png')
    image_resized = resize(image, (18,22), mode='reflect')
    img_array = []
    img_array.append(image_resized)
    grey = np.zeros((1, 18, 22))
    for image_num in range(len(img_array)):
            for rownum in range(len(img_array[image_num])):
                for colnum in range(len(img_array[image_num][rownum])):
                    grey[image_num][rownum][colnum] = weightedAverage(img_array[image_num][rownum][colnum])
    

    stopThread = threading.Thread(target=stopDetection)
    trafficThread = threading.Thread(target=trafficLightDetection)
    nnThread = threading.Thread(target=nnPrediction, args=(grey[0]))

    stopThread.start()
    trafficThread.start()
    nnThread.start()

    stopThread.join()
    trafficThread.join()
    nnThread.join()
    
    sendString = "STOP SIGN: " + str(stopPresent) + " Traffic Light: " + str(redPresent)
    print(pred_result)
    from time import sleep
    sleep(10)
    print(pred_result)
    
    c.send(str(sendString))
    c.close()

    #Neural Network Predition START
            
    
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")

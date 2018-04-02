from tkinter import Tk, Label, Button, Frame, Entry, Text, StringVar, filedialog
import tkinter.messagebox
import os
import tensorflow as tf
import DataPreProcessing
import numpy as np
from PIL import Image, ImageTk
from time import sleep
import threading
import time
import cv2
import re
from socket import *

class AutonomousDashboard:
    def __init__(self):
        #Create the main root window
        self.root = Tk()

        #Setup Variables
        self.currentdir = ""
        self.selectedModelDir = ""
        self.selectedTrafficDir = ""
        self.selectedStopDir = ""
        self.outputCounter = 1.0
        self.modelPath = StringVar()
        self.trafficCascadePath = StringVar()
        self.stopCascadePath = StringVar()

        #Set title of window
        self.root.title("Autonomous Vehcile Dashbaord")

        #Set Heading of the windows
        Label(self.root, text="Autonomous Vehicle Dashbaord", font="Helvetica 20 bold").grid(row=0, column=1,
                                                                                             columnspan=4, pady=15)

        ##Connection Settings
        self.connection_frame = Frame(self.root)

        Label(self.connection_frame, text="Connection Settings", font="Helvetica 18 bold").grid(row=0,column=0,
                                                                                                columnspan=3)
        Label(self.connection_frame, text="Client IP:").grid(row=1, column=0)
        Label(self.connection_frame, text="Client Port:").grid(row=2, column=0)

        self.txt_ip = Entry(self.connection_frame, width=20)
        self.txt_ip.grid(row=1, column=1, columnspan=2)
        self.txt_port = Entry(self.connection_frame, width=20)
        self.txt_port.grid(row=2, column=1, columnspan=2)

        self.connection_frame.grid(row=1, column=0, padx=20)
        ##

        ##Model Reloading
        self.reloading_frame = Frame(self.root)

        Label(self.reloading_frame, text="Model Reloading", font="Helvetica 18 bold").grid(row=0, column=2,
                                                                                           columnspan=4)
        Label(self.reloading_frame, text="Path to Model:").grid(row=1, column=0, columnspan=3, sticky="w")
        Label(self.reloading_frame, text="Path to STOP Cascade:").grid(row=2, column=0, columnspan=4, sticky="w")
        Label(self.reloading_frame, text="Path to Traffic Cascade:").grid(row=3, column=0, columnspan=4, sticky="w")

        self.txt_modelPath = Entry(self.reloading_frame, width=20, text=self.modelPath)
        self.txt_modelPath.grid(row=1, column=4, columnspan=3)
        self.txt_stopCascade = Entry(self.reloading_frame, width=20, text=self.stopCascadePath)
        self.txt_stopCascade.grid(row=2, column=4, columnspan=3)
        self.txt_trafficCascade = Entry(self.reloading_frame, width=20, text=self.trafficCascadePath)
        self.txt_trafficCascade.grid(row=3, column=4, columnspan=3)

        Button(self.reloading_frame, text="Browse", command=self.modelLoad).grid(row=1, column=7)
        Button(self.reloading_frame, text="Browse", command=self.stopLoad).grid(row=2, column=7)
        Button(self.reloading_frame, text="Browse", command=self.trafficLoad).grid(row=3, column=7)

        self.reloading_frame.grid(row=1, column=5, padx=20, pady=10)
        ##

        ##Start Button
        self.connect_frame = Frame(self.root)

        Button(self.connect_frame, text="Connect and Start Prediction", command=self.startServer).grid(row=0, column=0, columnspan=3)
        
        self.connect_frame.grid(row=6, column=4, pady=20)
        ##

        ##Model Results
        self.results_frame = Frame(self.root)

        Label(self.results_frame, text="Model Results", font="Helvetica 18 bold").grid(row=0, column=1, columnspan=2)
        Label(self.results_frame, text="Direction:").grid(row=1, column=0)
        Label(self.results_frame, text="STOP sign:").grid(row=1, column=1)
        Label(self.results_frame, text="Red Traffic:").grid(row=1, column=2)
        Label(self.results_frame, text="Green Traffic:").grid(row=1, column=3)

        Label(self.results_frame, text="").grid(row=2, column=0)
        Label(self.results_frame, text="").grid(row=2, column=1)
        Label(self.results_frame, text="").grid(row=2, column=2)
        Label(self.results_frame, text="").grid(row=2, column=3)
        
        self.results_frame.grid(row=7, column=5, pady=10)
        ##

        ##Vehicle Camera
        self.camera_frame = Frame(self.root)

        Label(self.camera_frame, text="Vehicle Camera:", font="Helvetica 18 bold").grid(row=0, column=1, columnspan=3)
        self.imageBox = Label(self.camera_frame)
        self.imageBox.grid(row=0, column=0)

        self.camera_frame.grid(row=7, column=0)
        ##

        ##Model Output
        self.outputFrame = Frame(self.root)

        Label(self.outputFrame, text="Raw Model Output", font="Helvetica 18 bold").grid(row=0, column=0, columnspan=2)
        self.modelOutput = Text(self.outputFrame, height=10, width=40)
        self.modelOutput.grid(row=1, column=0)

        self.outputFrame.grid(row=10, column=5, columnspan=4, pady=10)
        ##
        
        self.root.mainloop()

    def modelLoad(self):
        self.currentdir = os.getcwd()
        self.selectedModelDir = filedialog.askopenfilename(parent=self.root, initialdir=self.currentdir,
                                                           title="Please select a file:")
        self.modelPath.set(self.selectedModelDir)

    def stopLoad(self):
        self.currentdir = os.getcwd()
        self.selectedStopDir = filedialog.askopenfilename(parent=self.root, initialdir=self.currentdir,
                                                          title="Please select a file")
        self.stopCascadePath.set(self.selectedStopDir)

    def trafficLoad(self):
        self.currentdir = os.getcwd()
        self.selectedTrafficDir = filedialog.askopenfilename(parent=self.root, initialdir=self.currentdir,
                                                             title="Please select a file")
        self.trafficCascadePath.set(self.selectedTrafficDir)

    def outputToText(self, text):
        self.modelOutput.insert(str(self.outputCounter), text)
        self.outputCounter += 1.0
        

    def checkIP(self, providedIP):
        pat = re.compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        checker = pat.match(providedIP)
        if checker:
            self.outputToText("Valid IP")
        else:
            tkinter.messagebox.showinfo('Invalid IP', 'Invalid Ip')


    def startServer(self):
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
            self.outputToText("[*] Recieving...")
            l = conn.recv(150000)
            while (l):
                f.write(l)
                l = conn.recv(150000)
            self.outputToText("[*] Image Successfully Recieved")

        #Initalise a new connection to the server
        def init_new_conn():
            s = socket(AF_INET, SOCK_STREAM)
            s.bind((str(self.txt_ip.get()), int(self.txt_port.get())))
            self.outputToText("[*] Waiting for connection...")
            s.listen(5)
            return s

        #Establish a link with client
        def listen_new_conn(s):
            conn, a = s.accept()
            self.outputToText("[*] Connected to Client!")
            return conn

        #TCP Send Results
        def send_results(directionResult, trafficResult, stopResult, dest):
            completeResult = str(directionResult) + str(trafficResult) + str(stopResult)
            self.outputToText("[*] Sending", completeResult)
            completeResult = str.encode(completeResult)
            dest.send(completeResult)
            self.outputToText("[*] Result sent")

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
                self.outputToText("[*] Model Restored")
                result = sess.run(tf.argmax(pred, 1), feed_dict={x: recv_image})
                self.outputToText("[*] Model prediction result: ", str(result))
                pred_result = result

        #STOP Sign Detection
        def stopDetection():
            global stopPresent, stop_sign_active, stop_flag, stop_start, stop_finish, stop_time, drive_time_after_stop
            stopCascade = cv2.CascadeClassifier('stop_class.xml')
            stop = stopCascade.detectMultiScale(grey,
                                                scaleFactor=1.1,
                                                minNeighbors=2
                                                )
            for (x, y, w, h) in stop:
                #Calculate Distance
                stop_distance = (stop_actualWidth * focal) / w
                self.outputToText("Distance:", stop_distance)

                #If distance is less than 25cm
                if 0 < stop_distance < 30 and stop_sign_active:
                    self.outputToText("STOP SIGN AHEAD")
                    stopPresent = "T"

                    #Storing the first time to stop the vehicle
                    if stop_flag is False:
                        stop_start = cv2.getTickCount()
                        stop_flag = True

                    #Each loop a new time is recorded to compare against start time
                    stop_finish = cv2.getTickCount()
                    stop_time = (stop_finish - stop_start) / cv2.getTickFrequency()
                    self.outputToText("STOP Halted for:", stop_time)

                    #If 5 seconds are up, begin to drive and set ignore flags
                    if stop_time > 5:
                        self.outputToText("5 Seconds up, continue driving...")
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

            self.outputToText("[*] STOP Detection Complete")
            self.outputToText("[*] STOP sign present:", stopPresent)

        def trafficLightDetection():
            #Not working well - replace with y + (height / 2) then the usual greater and less than
            global lightColor
            trafficCascade = cv2.CascadeClassifier('traffic.xml')
            traffic = trafficCascade.detectMultiScale(grey, 1.1, 2)

            for (x,y,w,h) in traffic:
                self.outputToText("[*] Detected Traffic Light")
                roi_gray = grey[y:y+h-20, x:x+w]

                #Blur to remove any noise
                blurred = cv2.GaussianBlur(roi_gray, (41, 41), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)

                #Coordinates of the brightest light
                print(maxLoc[1])

                #Get Distance
                traffic_distance = (traffic_actualWidth * focal) / w
                self.outputToText("Distance:", traffic_distance)

                #If close enough - Identify the light color
                if 0 < traffic_distance < 40:
                    if maxLoc[1] >= 40:#used to be 35
                        self.outputToText("[*] Green Detected")
                        lightColor = "G"
                    elif maxLoc[1] <= 40:#used to br 30
                        self.outputToText("[*] Red Detected")
                        lightColor = "R"
                else:
                    self.outputToText("[*] Traffic Light Not Close Enough")
                    lightColor = "N"

                self.outputToText("[*] Traffic Light Detected")
            self.outputToText("[*] Traffic Light Detection Complete")


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

                b,g,r = cv2.split(img)
                display_img = cv2.merge((r,g,b))
                im = Image.fromarray(display_img)
                imgtk = ImageTk.PhotoImage(image=im)

                self.imageBox.config(image=imgtk)

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
            self.outputToText("[*] Connection being closed...")
        finally:
            conn.close()

            
        

interface = AutonomousDashboard()

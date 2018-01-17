from socket import *
import numpy as np
import matplotlib.pyplot as plt
import DataPreProcessing
import threading
import tensorflow as tf

#Resul variables initalisation
stopPresent = False
greenPresent = False
redPresent = False
pred_result = [0]

#TCP Recieve method
def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]
    print('[*] Successfully Received Image!')

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
    stop = stopCascade.detectMultiScale(img,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30,30)
                                        )
    for (x, y, w, h) in stop:
        stopPresent = True
    print("STOP Detection Complete")


#Initalising the TCP Connection
s = socket(AF_INET, SOCK_STREAM)
s.bind(('localhost', 25000))
print("[*] Waiting for connection...")
s.listen(1)
conn, a = s.accept()
print("[*] Connected to client")

try:
    while True:
        #Creating the recieving array
        recieved_array = np.zeros((36, 44, 3))
        recv_into(recieved_array, conn)
        print("[*] Image Array Recieved")

        #Image Greyscale conversion
        img_converter = DataPreProcessing.DatasetProcessing()
        img = img_converter.getSingleImage(recieved_array)

        #Initalise threads
        directionThread = threading.Thread(target=directionPrediction, args=(img))

        #Start the threads
        directionThread.start()

        #Join the queue - so they all wait to finish before sending the result
        directionThread.join()

except KeyboardInterrupt:
    print("[*] Connection being closed...")
finally:
    conn.close()

import tensorflow as tf
from scipy import misc
from skimage.transform import resize
import numpy as np
import DataPreProcessing

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

getimg_1 = DataPreProcessing.DatasetProcessing(None, 1, "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Forward/forward (653).png")
getimg_2 = DataPreProcessing.DatasetProcessing(None, 1, "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Forward/forward (1353).png")
getimg_3 = DataPreProcessing.DatasetProcessing(None, 1, "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Left/left (2986).png")
getimg_4 = DataPreProcessing.DatasetProcessing(None, 1, "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Left/left (2565).png")
getimg_5 = DataPreProcessing.DatasetProcessing(None, 1, "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Right/right (340).png")
getimg_6 = DataPreProcessing.DatasetProcessing(None, 1, "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Right/right (980).png")

img_1 = getimg_1.getSingleImage()
img_2 = getimg_2.getSingleImage()
img_3 = getimg_3.getSingleImage()
img_4 = getimg_4.getSingleImage()
img_5 = getimg_5.getSingleImage()
img_6 = getimg_6.getSingleImage()

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
y = tf.placeholder('float', [None, n_classes])
pred = multilayer_perceptron(x, weights, biases)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "model.ckpt")
    print("[*] MODEL RESTORED SUCCESSFULLY")
    result = sess.run(tf.argmax(pred, 1), feed_dict={x: img_1})
    print "FORWARD IMAGE 653: ", result[0]
    result2 = sess.run(tf.argmax(pred, 1), feed_dict={x: img_2})
    print "FORWARD IMAGE 1353: ", result2[0]
    result3 = sess.run(tf.argmax(pred, 1), feed_dict={x: img_3})
    print "LEFT IMAGE 2986: ", result3[0]
    result4 = sess.run(tf.argmax(pred, 1), feed_dict={x: img_4})
    print "LEFT IMAGE 2565: ", result4[0]
    result5 = sess.run(tf.argmax(pred, 1), feed_dict={x: img_5})
    print "RIGHT IMAGE 340: ", result5[0]
    result6 = sess.run(tf.argmax(pred, 1), feed_dict={x: img_6})
    print "RIGHT IMAGE 980: ", result6[0]

import DataPreProcessing
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

##DATASET PREPERATION##

#Instances of DataPreProcessing and retreival of image and label arrays
#Forward
f_images = DataPreProcessing.DatasetProcessing("forward", 3000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Forward/")
f_images.generateDataset()
f_array = f_images.getImgArray()
f_lbl = f_images.getLblArray()

#Left
l_images = DataPreProcessing.DatasetProcessing("left", 3000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Left/")
l_images.generateDataset()
l_array = l_images.getImgArray()
l_lbl = l_images.getLblArray()

#Right
r_images = DataPreProcessing.DatasetProcessing("right", 3000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Right/")
r_images.generateDataset()
r_array = r_images.getImgArray()
r_lbl = r_images.getLblArray()

#Concatenate all image and label arrays
all_images_array = np.concatenate((f_array, l_array, r_array), axis=0)
all_labels_array = np.concatenate((f_lbl, l_lbl, r_lbl), axis=0)

#Reshuffle the new arrays
image_array, image_labels = shuffle(all_images_array, all_labels_array, random_state=9000)

##END DATASET PREPERATION##

##DEFINING PARAMETERS##

#Defining model parameters
learning_rate = 0.0001
training_epochs = 25
batch_size = 50
n_classes = 3
n_samples = 9000
n_input = 396
n_hidden_1 = 256
n_hidden_2 = 256

#Defining weights
weights = {
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
#Defining biases
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

##END DEFINING PARAMETERS##

##DEFNING THE NEURAL NETWORK FUNCTION##

#Neural network function
def multilayer_perceptron(x, weights, biases):
    '''
    x: placeholder for data input
    weights: dict of weights
    biases: dict of bias values
    '''
    #HIDDEN LAYER 1  (X * W) + B
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #RELU ACTIVATION FUNCTION  ((X * W) + B) --> f(x) = max(0, x)
    layer_1 = tf.nn.relu(layer_1)

    #HIDDEN LAYER 3
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #RELU ACTIVATION FUNCTION
    layer_2 = tf.nn.relu(layer_2)

    #OUTPUT LAYER
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

#Defining placeholders
'''
x - same as the number of inputs
y - output - same as classes
'''
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

#Setting up the model
pred = multilayer_perceptron(x, weights, biases)

#Define cost and optimisation functions
cost = tf.reduce_mean(tf.nn.softmax_corss_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

##END DEFINING MODEL PARAMETERS##

##STARTING TRAINING OF MODEL##

#Initialise the session, variables and model saver method
sess = tf.InteractiveSession()
init = tf.global_variables_initialiser()
saver = tf.train.Saver()
sess.run(init)

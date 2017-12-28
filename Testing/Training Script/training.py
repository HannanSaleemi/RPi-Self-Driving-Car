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
training_epochs = 20
batch_size = 0.4
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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

##END DEFINING MODEL PARAMETERS##

##STARTING TRAINING OF MODEL##

#Initialise the session, variables and model saver method
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

#Traing the model on the images
start_batch = 0
end_batch = batch_size

#Main training loop (epoch - one time over each image)
for epoch in range(training_epochs):

    #Resetting the average for each epoch
    avg_cost = 0

    #Defining the total batch
    total_batch = int(n_samples/batch_size)

    #For each batch of images
    for i in range(total_batch):

        #Get the next batch
        batch_x = image_array[start_batch:end_batch]
        batch_y = image_labels[start_batch:end_batch]

        #Optimisation and loss
        _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})

        avg_cost += c/total_batch

        #Get the next batch start and end values
        start_batch += batch_size
        end_batch += batch_size

    #Reset batch values
    start_batch = 0
    end_batch = batch_size

    #Output epoch and cost - helps to see if model is improving in accuracy over epochs
    print("Epoch {} Cost {:.4f}".format(epoch+1, avg_cost))
print("Model has completed {} epochs of training".format(training_epochs))

#Saving the model to file
save_path = saver.save(sess, "model.ckpt")
print("[*] Model saved to file")

##END TRAINING##

##MODEL EVALUATION##

#Training set accuracy
correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
correct_predictions = tf.cast(correct_predictions, 'float')
accuracy = tf.reduce_mean(correct_predictions)
print("Training Accuracy: ", accuracy.eval({x: image_array, y: image_labels}))

#Testing set accuracy
#Forward images and labels
test_f_images = DataPreProcessing.DatasetProcessing("forward", 1000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Testing/Forward/")
test_f_images.generateDataset()
test_f_array = test_f_images.getImgArray()
test_f_lbl = test_f_images.getLblArray()

#Left images and labels
test_l_images = DataPreProcessing.DatasetProcessing("left", 1000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Testing/Left/")
test_l_images.generateDataset()
test_l_array = test_l_images.getImgArray()
test_l_lbl = test_l_images.getLblArray()

#Right image and labels
test_r_images = DataPreProcessing.DatasetProcessing("right", 1000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Testing/Right/")
test_r_images.generateDataset()
test_r_array = test_r_images.getImgArray()
test_r_lbl = test_r_images.getLblArray()

#Concatenate images and labels
test_images = np.concatenate((test_f_array, test_l_array, test_r_array), axis=0)
test_labels = np.concatenate((test_f_lbl, test_l_lbl, test_r_lbl))

#Shuffle images and labels
test_images_array, test_labels_array = shuffle(test_images, test_labels, random_state=9000)

#Getting the accuracy
test_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
test_predictions = tf.cast(test_predictions, 'float')
test_accuracy = tf.reduce_mean(test_predictions)
print("Testing Accuracy: ", test_accuracy.eval({x: test_images_array, y: test_labels_array}))

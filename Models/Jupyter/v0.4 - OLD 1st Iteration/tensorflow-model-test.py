import tensorflow as tf
from scipy import misc
from skimage.transform import resize
import numpy as np

##[0] = forward
##[1] = left
##[2] = right


def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

image = misc.imread('/Volumes/TRANSCEND/SDC/JupyterNotebookExamples/cardataset/training_set/Right/trainframe (337).png')
image_resized = resize(image, (18,22), mode='reflect')
print(image_resized.shape)
img_array = []
img_array.append(image_resized)
grey = np.zeros((1, 18, 22))
for image_num in range(len(img_array)):
        for rownum in range(len(img_array[image_num])):
            for colnum in range(len(img_array[image_num][rownum])):
                grey[image_num][rownum][colnum] = weightedAverage(img_array[image_num][rownum][colnum])



def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

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
array_test = grey[0].reshape(1,396)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "model.ckpt")
    print("Model Restored")
    result = sess.run(tf.argmax(pred, 1), feed_dict={x: array_test})
    print(result)
    pred_result = result

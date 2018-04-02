from tkinter import Tk, Label, Button, Frame, Entry, Text, filedialog, StringVar, DoubleVar
import tkinter.messagebox
import os
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import DataPreProcessing

class ModelCreationInterface:
    def __init__(self):
        #Create the main root window
        self.root = Tk()

        #Setup Variables needed
        self.currentdir = ""
        self.selecteddir = ""
        self.displayChosen = StringVar()
        self.trainAccuracy = StringVar()
        self.testAccuracy = StringVar()
        self.trainAccuracy.set("0")
        self.testAccuracy.set("0")
        self.outputCounter = 1.0
        
        #Set title of the window, heading and help button
        self.root.title("Training - Model Creation")
        Label(self.root, text="Training - Model Creation", font="Helvetica 18 bold").grid(row=0, column=1,
                                                                                        columnspan=3,
                                                                                        pady=15)
        self.helpButton = Button(self.root, text="Help", command=self.helpMessage).grid(row=0, column=4,
                                                                                        sticky="e",
                                                                                        padx=15)
        
        ##Widgets for Dataset Configuration
        self.data_config_frame = Frame(self.root)
        
        Label(self.data_config_frame, text="Dataset Configuration",
              font="Helvetica 18 bold").grid(row=0, column=0, columnspan=3)
        
        Label(self.data_config_frame, text="Dataset Path").grid(row=1, column=1)
        self.txt_datasetPath = Entry(self.data_config_frame, width=20, text=self.displayChosen).grid(row=2, column=0, columnspan=2)
        Button(self.data_config_frame, text="Browse", command=self.browseDirectory).grid(row=2, column=2)

        Label(self.data_config_frame, text="Image Size").grid(row=3, column=0)
        self.txt_imageWidth = Entry(self.data_config_frame, width=5)
        self.txt_imageWidth.grid(row=3, column=1)
        self.txt_imageHeight = Entry(self.data_config_frame, width=5)
        self.txt_imageHeight.grid(row=3, column=2)

        Label(self.data_config_frame, text="Model Name").grid(row=4, column=0)
        self.txt_modelName = Entry(self.data_config_frame, width=20)
        self.txt_modelName.grid(row=4, column=1, columnspan=2)
        #Draw Frame
        self.data_config_frame.grid(row=1, column=0, rowspan=5, columnspan=3, padx=10, pady=10)
        ####
        
        ##Widget for Model Configuration
        self.model_frame = Frame(self.root)

        Label(self.model_frame, text="Model Configuration", font="Helvetica 18 bold").grid(row=0,
                                                                                          column=0,
                                                                                          columnspan=4)
        Label(self.model_frame, text="Learning Rate").grid(row=1, column=0, columnspan=2, sticky="w")
        Label(self.model_frame, text="Epochs").grid(row=2, column=0, sticky="w")
        Label(self.model_frame, text="Batch Size").grid(row=3, column=0, columnspan=2, sticky="w")

        self.txt_learningRate = Entry(self.model_frame, width=20)
        self.txt_learningRate.grid(row=1, column=2, columnspan=2)
        self.txt_epochs = Entry(self.model_frame, width=20)
        self.txt_epochs.grid(row=2, column=2, columnspan=2)
        self.txt_batchSize = Entry(self.model_frame, width=20)
        self.txt_batchSize.grid(row=3, column=2, columnspan=2)
        
        self.model_frame.grid(row=1, column=4, pady=10, padx=10)
        ####
        
        ##Start training button
        self.start_frame = Frame(self.root)
        Button(self.start_frame, text="Start Training", command=self.beginTraining).grid(row=0, column=0, pady=20)
        self.start_frame.grid(row=6, column=3)
        ####
        
        ##Training Output
        self.output_frame = Frame(self.root)
        Label(self.output_frame, text="Training Outputs", font="Helvetica 18 bold").grid(row=0, column=0,
                                                                                   columnspan=2)
        self.trainingOutput = Text(self.output_frame, height=10, width=40)
        self.trainingOutput.grid(row=1, column=1)
        self.output_frame.grid(row=8, column=0, padx=10, pady=20)
        ####
        
        ##Model Accraucy Stats
        self.accuracy_frame = Frame(self.root)

        Label(self.accuracy_frame, text="Model Accuracy Statistics", font="Helvetica 18 bold").grid(row=0,
                                                                                                    column=0,
                                                                                                    columnspan=3)
        Label(self.accuracy_frame, text="Final Reported Accuracy").grid(row=1, column=0, columnspan=2, sticky="w")
        Label(self.accuracy_frame, text="Testing Data Accuracy").grid(row=2, column=0, columnspan=2, sticky="w")
        
        self.lbl_trainAccuracy = Label(self.accuracy_frame, textvariable=self.trainAccuracy).grid(row=1, column=2, columnspan=2)
        self.lbl_testAccuracy = Label(self.accuracy_frame, textvariable=self.testAccuracy).grid(row=2, column=2, columnspan=2)
        
        self.accuracy_frame.grid(row=8, column=4, padx=10)
        

        #Enter mainloop and show window
        self.root.mainloop()

    def helpMessage(self):
        #Display dialog box for information:
        tkinter.messagebox.showinfo('Help', \
                                '''Dataset Path - Must include a Training and Testing folder,each with
Forward,
Left and
Right Image Folders.
Image Size need a width (first box) and height (second box).
Model needs a name to save in the same directory as this python file.
Learning rate needs to be a decimal between 0 and 1.
Epochs needs to be an integer.
Batch Size needs to be an integer.
Once you press start training, you will see the output and accuracy.''')

    def browseDirectory(self):
        self.currentdir = os.getcwd()
        self.selecteddir = filedialog.askdirectory(parent=self.root, initialdir=self.selecteddir,
                                                   title='Please seletc a directory')
        print("Directory Selected: " + self.selecteddir)
        print(self.selecteddir+"/Training/Forward/")
        self.displayChosen.set(self.selecteddir)

    def outputToText(self, text):
        self.trainingOutput.insert(str(self.outputCounter), text)
        self.outputCounter += 1


    def beginTraining(self):
        try:
            if len(self.selecteddir) <= 0:
                tkinter.messagebox.showinfo('Invalid Path', 'Please select a valid directory')
            float(self.txt_learningRate.get())
            int(self.txt_imageWidth.get())
            int(self.txt_imageHeight.get())
            int(self.txt_epochs.get())
            int(self.txt_batchSize.get())
        except Exception as e:
            tkinter.messagebox.showinfo('Invalid Entry', e)

        try:
            #Forward Images
            f_images = DataPreProcessing.DatasetProcessing("forward", 3000,
                        self.selecteddir+"/Training/Forward/")
            f_images.generateDataset()
            f_array = f_images.getImgArray()            
            f_lbl = f_images.getLblArray()

            #Left Images
            l_images = DataPreProcessing.DatasetProcessing("left", 3000,
                        self.selecteddir+"/Training/Left/")
            l_images.generateDataset()
            l_array = l_images.getImgArray()
            l_lbl = l_images.getLblArray()

            #Right Images
            r_images = DataPreProcessing.DatasetProcessing("right", 3000,
                    self.selecteddir+"/Training/Right/")
            r_images.generateDataset()
            r_array = r_images.getImgArray()
            r_lbl = r_images.getLblArray()
        except Exception as e:
            tkinter.messagebox.showinfo('Invaid Directory', 'Directory may be invalid or folder structure incorrect')

        #Concatenate all image and label arrays
        all_images_array = np.concatenate((f_array, l_array, r_array), axis=0)
        all_labels_array = np.concatenate((f_lbl, l_lbl, r_lbl), axis=0)

        #Reshuffle the new arrays
        image_array, image_labels = shuffle(all_images_array, all_labels_array, random_state=9000)

        ##END DATASET PREPERATION##

        ##DEFINING PARAMETERS##

        #Defining model parameters
        learning_rate = float(self.txt_learningRate.get())
        training_epochs = int(self.txt_epochs.get())
        batch_size = int(self.txt_batchSize.get())
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

        #Placeholders
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
            self.outputToText("Epoch {} Cost {:.4f}".format(epoch+1, avg_cost))
        self.outputToText("Model has completed {} epochs of training".format(training_epochs))

        #Saving the model to file
        save_path = saver.save(sess, self.selecteddir+"/"+self.txt_modelName.get()+".ckpt")
        self.outputToText("[*] Model saved to file")

        ##END TRAINING##

        ##MODEL EVALUATION##

        #Training set accuracy
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        correct_predictions = tf.cast(correct_predictions, 'float')
        accuracy = tf.reduce_mean(correct_predictions)
        print(accuracy.eval({x: image_array, y: image_labels}))
        self.trainAccuracy.set(str(accuracy.eval({x: image_array, y: image_labels})))

        #Testing set accuracy
        #Forward images and labels
        test_f_images = DataPreProcessing.DatasetProcessing("forward", 1000,
                self.selecteddir+"/Testing/Forward/")
        test_f_images.generateDataset()
        test_f_array = test_f_images.getImgArray()
        test_f_lbl = test_f_images.getLblArray()

        #Left images and labels
        test_l_images = DataPreProcessing.DatasetProcessing("left", 1000,
                self.selecteddir+"/Testing/Left/")
        test_l_images.generateDataset()
        test_l_array = test_l_images.getImgArray()
        test_l_lbl = test_l_images.getLblArray()

        #Right image and labels
        test_r_images = DataPreProcessing.DatasetProcessing("right", 1000,
                self.selecteddir+"/Testing/Right/")
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
        print(test_accuracy.eval({x: test_images_array, y: test_labels_array}))
        self.testAccuracy.set(str(test_accuracy.eval({x: test_images_array, y: test_labels_array})))
           

interface = ModelCreationInterface()

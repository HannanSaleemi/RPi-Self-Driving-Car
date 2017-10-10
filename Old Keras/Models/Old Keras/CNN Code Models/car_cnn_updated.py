# Convolutional Neural Networks for Self-Driving Car

#Start with just 1 convolutional layer and pooling layer and add on to imporve

#Things that can be adjusted in order to imporve accuracy:
# Dataset balance
# Amount of Convolutional layers and pooling layers
# Amount of fully connected layers
# Number of nodes/neurons in fullt connected layer (32 seemed optimal for ryan and zheng)
# More to come...

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow VIA Website and follow instructions

# Installing Keras
# pip install --upgrade keras

# Data needs to be generated
# For a system like this, we will need each frame from camera to be processed into CNN
# Usually a 70% Training and 30% Testing dataset works correctly, I'll try this
# Imperial Uni Student used 60% Training set and 40% Testing set
# Recommended - TRAINING: 3000 L, 3000 R, 3000 F    75%
#             - TESTING: 1000 L, 1000 R, 1000 F     25%
# TOTAL - 12,000

#SUGGESTED STARTING STRUCTURE:
#Convoltuional Layer
#Pooling Layer
#Flattening Layer
#Fully Connected Layer
#Output Layer

## Importing Important Keras Packages
from keras.models import Sequential                              #Used to initialise the NN
from keras.layers import Convolution2D                           #Used for the convolutional layer
from keras.layers import MaxPooling2D                            #Used for the pooling layer
from keras.layers import Flatten                                 #Used for the Flattening layer
from keras.layers import Dense                                   #USed for Fully connected layers
from keras.layers import Dropout

#Initalising the CNN Model
classifier = Sequential()                           #Creating an object of the Sequential class

## CONVOLUTIONAL LAYER 1
classifier.add(Convolution2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))

## POOLING LAYER 1
classifier.add(MaxPooling2D(pool_size=(2,2)))

#################TEST OUT SIMPLE NETWORK AND TRY ALL OTHER COMBINATIONS AFTERWARDS############################
## CONVOLUTIONAL LAYER 2
classifier.add(Convolution2D(32, (3,3), activation = 'relu'))

## POOLING LAYER 2
classifier.add(MaxPooling2D(pool_size=(2,2)))

##CONVOLUTIONAL LAYER 3
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))

## POOLING LAYER 3
classifier.add(MaxPooling2D(pool_size=(2,2)))

#################TEST OUT SIMPLE NETWORK AND TRY ALL OTHER COMBINATIONS AFTERWARDS############################

## FLATTENING LAYER
classifier.add(Flatten())

## FULLY CONNECTED LAYER 1
classifier.add(Dense(units=32, activation='relu'))          # 32 worked well for ryan and zheng
classifier.add(Dropout(0.2))                                 # PREVENTS OVERFITTING

## FULLY CONNECTED LAYER 2
classifier.add(Dense(units=32, activation='relu'))

## FULLY CONNECTED LAYER 3
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dropout(0.1)) 

## OUPUT LAYER
classifier.add(Dense(units=3, activation='softmax'))

## COMPILING THE CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

## FITTING DATA TO CNN
from keras.preprocessing.image import ImageDataGenerator

#Randomly applying transformations to the test data set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#Rescaling the pixels of test set so that they have values between 0 and 1
test_datagen = ImageDataGenerator(rescale=1./255)

#Point to trainig set directory
#RESIZING IMAGES TO 64X64 TO FIT CNN AND TELLING WHERE TRAINING SET IS
training_set= train_datagen.flow_from_directory('C:/Users/Hannan Saleemi/Desktop/Self-Driving Car/cardataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

#RESIZING IMAGES TO 64X64 TO FIT CNN AND TELLING WHERE TEST SET IS
test_set = test_datagen.flow_from_directory('C:/Users/Hannan Saleemi/Desktop/Self-Driving Car/cardataset/testing_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

## RUNNING THE TRAINING ON THE MODEL
classifier.fit_generator(training_set,
                         steps_per_epoch=9000,       #9000 samples in training set
                         epochs=10,                  #How many times to pass the whole set through
                         validation_data=test_set,   #Testing data
                         validation_steps=3000)      #3000 samples in testing set
                  
## SAVING THE MODEL TO FILE                         
classifier.save('car_cnn.h5')

print("ALL DONE - TRAINED CNN STORED IN OUTPUT - ALL SUCCESSFULL")

## DELETE THE MODEL
#del classifier

## IMPORT THE MODEL
#from keras.models import load_model
#classifier = load_model('Car_trained_model_1.h5')

## MAKING NEW PREDICTIONS

#import numpy as np
#from keras.preprocessing import image

## LOAD TESTING IMAGE - REMEMBER TO CHANGE TARGET SIZE TO FIT CNN
#test_image = image.load_img('dataset/testing_set/Left/trainframe (579).png', target_size=(64, 64))

## TURN IMAGE INTO ARRAY
#test_image = image.img_to_array(test_image)

## ADD AN EXTRA DIMENSION TO AVOID ERRORS
#test_image = np.expand_dims(test_image, axis=0)

## MAKE PREDICTIONS
#result = classifier.predict(test_image)


#results = [, Forward, Right]
#training_set.class_indicies














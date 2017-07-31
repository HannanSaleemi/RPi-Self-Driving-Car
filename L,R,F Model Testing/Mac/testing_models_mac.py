## IMPORT THE MODEL
from keras.models import load_model
classifier = load_model("/Users/intern.mac/Desktop/RPi-Self-Driving-Car/Trained Models/28-july-car_cnn.h5")

## MAKING NEW PREDICTIONS

import numpy as np
from keras.preprocessing import image

## LOAD TESTING IMAGE - REMEMBER TO CHANGE TARGET SIZE TO FIT CNN
test_image = image.load_img('/Users/intern.mac/Desktop/RPi-Self-Driving-Car/cardataset/testing_set/Right/trainframe (5).png', target_size=(64, 64))

## TURN IMAGE INTO ARRAY
test_image = image.img_to_array(test_image)

## ADD AN EXTRA DIMENSION TO AVOID ERRORS
test_image = np.expand_dims(test_image, axis=0)

## MAKE PREDICTIONS
result = classifier.predict(test_image)


#results = [, Forward, Right]
#training_set.class_indicies

#[forward,left,right]
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

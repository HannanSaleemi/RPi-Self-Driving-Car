import numpy as np
import glob
from scipy import misc
from skimage.transform import resize
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class DatasetProcessing(object):

    #Initaliser method
    def __init__(self, typeOfImg, numofimgs, pathtoimgs):
        self.pathToImgs = pathtoimgs
        self.numOfImgs = numofimgs
        self.typeOfImages = typeOfImg
        self.image_list = []
        self.image_shape = [18, 22]
        self.grey = np.zeros((self.numOfImgs, self.image_shape[0], self.image_shape[1]))
        self.image_labels = np.zeros(shape=(self.numOfImgs, 3))
        self.flatten_size = (self.image_shape[0]) * (self.image_shape[1])

    #Import and Resize the images
    def importAndResize(self):
        path = self.pathToImgs
        for filename in glob.glob(path+'*.png'):
            image = misc.imread(filename)
            image_resized = resize(image, (18,22), mode='reflect')
            self.image_list.append(image_resized)
        print("STAGE 1 COMPLETE")
        self.convertToGreyscale()

    #Weighted Average Conversion
    def weightedAverage(self, pixel):
        return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

    #Converting each image from color to greyscale
    def convertToGreyscale(self):
        for image_num in range(len(self.image_list)):
            for rownum in range(len(self.image_list[image_num])):
                for colnum in range(len(self.image_list[image_num][rownum])):
                    self.grey[image_num][rownum][colnum] = self.weightedAverage(self.image_list[image_num][rownum][colnum])
        print("STAGE 2 COMPLETE")
        self.flattenImages()

    #Flatten the images
    def flattenImages(self):
        self.image_list = self.grey.reshape(self.numOfImgs, self.flatten_size)
        print("STAGE 3 COMPLETE")
        self.createLabels()

    #Creating the labels for the images
    def createLabels(self):
        if self.typeOfImages.lower() == "forward":
            label = 0
        elif self.typeOfImages.lower() == "left":
            label = 1
        elif self.typeOfImages.lower() == "right":
            label = 2
        for i in range(self.numOfImgs):
            self.image_labels[i][label] = float(1)
        print("STAGE 4 COMPLETE")
        self.shuffleImages()

    #Shuffle the images and the labels
    def shuffleImages(self):
        print("BEFORE SHUFFLE:")
        print(self.image_list[0])
        print(self.image_labels[0])
        self.image_list, self.image_labels = shuffle(self.image_list, self.image_labels, random_state = 4000)
        print("AFTER SHUFFLE:")
        print(self.image_list[0])
        print(self.image_labels[0])
        print("STAGE 5 COMPLETE")

    #Starting the dataset processing process
    def generateDataset(self):
        print("[*] STARTING DATASET PROCESSING")
        self.importAndResize()
        print("[*] DATASET PROCESSING COMPLETE")

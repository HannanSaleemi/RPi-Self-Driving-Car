### DATA-PREPROCESSING CLASS ###

####### NOTES/TODO #######
# - Change to remove global and add self to all global variables
# - Continue with the program

# [START IMPORTS] #
import numpy as np
import glob
from scipy import misc
from skimage.transform import resize
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# [END IMPORTS] #

class DatasetProcessing(object):

    #Initialiser class
    def __init__(self, typeOfImg, numofimgs, pathtoimg):
        self.pathToImgs = pathtoimg
        self.numOfImgs = numofimgs
        self.typeOfImages = typeOfImg
        self.image_list = []
        self.image_shape = [18,22]
        self.grey = np.zeros((self.numOfImgs, self.image_shape[0], self.image_shape[1]))
        self.image_labels = np.zeros(shape=(self.numOfImgs, 3))
        self.flatten_size = (self.image_shape[0])*(self.image_shape[1])

    #Generating the dataset
    def generateDataset(self):
        print("Starting...")
        self.importAndResize()
        print("Done...")

    #Greyscale Conversion method
    def weightedAverage(self, pixel):
        return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

    #Import and Resize
    def importAndResize(self):
        img_count = 0
        path = self.pathToImgs
        for filename in glob.glob(path+'*.png'):
            img_count += 1
            image = misc.imread(path+'trainframe ('+str(img_count)+').png')
            image_resized = resize(image, (18,22), mode='reflect')
            self.image_list.append(image_resized)
        
        print("STAGE 1 COMPLETE")
        print("IMAGE COUNT", str(img_count))
        print("Length of image array:", str(len(self.image_list)))
        self.convertToGreyscale()


    #Converting to Greyscale
    def convertToGreyscale(self):
        for image_num in range(len(self.image_list)):
            for rownum in range(len(self.image_list[image_num])):
                for colnum in range(len(self.image_list[image_num][rownum])):
                    self.grey[image_num][rownum][colnum] = self.weightedAverage(self.image_list[image_num][rownum][colnum])
            #Output Logging - Testing Only
            #if image_num % 100 = 0:
            #    print("Finished Converting image #", (image_num))
        print("STAGE 2 COMPLETE")
        print("length of grey:", str(len(self.grey)))
        self.flattenImages()

    #Reshaping / Flattening
    def flattenImages(self):
        self.image_list = self.grey.reshape(self.numOfImgs,self.flatten_size)
        print("STAGE 3 COMPLETE")
        print("SIZE of image_list", str(self.image_list.shape))
        self.createLabels()

    #Create Labels for the images
    def createLabels(self):
        if self.typeOfImages.lower() == "forward":
            label = 0
        elif self.typeOfImages.lower() == "left":
            label = 1
        elif self.typeOfImages.lower() == "right":
            label = 2
        for i in range(self.numOfImgs):
            self.image_labels[i][label] = float(1)
        self.shuffleImages()
        print("STAGE 4 COMPLETE")
        print("LABELS CREATED")


    #Shuffle images and labels
    def shuffleImages(self):
        self.image_list, self.image_labels = shuffle(self.image_list, self.image_labels, random_state = 4000)
        print("STAGE 5 COMPLETE")
        print("Images shuffled")
        print("Length of image array", str(len(self.image_list)))
        
    def showImg(self, img_num):
        image = self.image_list[img_num].reshape(1,396)
        print(image.shape)
        image = self.image_list[img_num].reshape(18,22)
        print(image.shape)
        plt.imshow(image, cmap='Greys')
        plt.show()
        
    def getImgArray(self):
        return self.image_list
    
    def getLblArray(self):
        return self.image_labels
        












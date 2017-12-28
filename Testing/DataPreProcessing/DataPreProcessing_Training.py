import DataPreProcessing

#Creating the instance
forward = DataPreProcessing.DatasetProcessing(-4, 3000,
        "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Right/")

#Calling the starter method
forward.generateDataset()

#Retreiving the image and labels array
i_array = forward.getImgArray()
l_array = forward.getLblArray

print(i_array[1])



















#Greyscale Conversion
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.transform import resize

image = misc.imread("/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Forward/forward (1).png")
image_resized = resize(image, (18,22), mode='reflect')
image_list = []
image_list.append(image_resized)
grey = np.zeros((1, 18, 22))

#Weighted Average Conversion
def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

#Converting each image from color to greyscale
for image_num in range(len(image_list)):
    for rownum in range(len(image_list[image_num])):
        for colnum in range(len(image_list[image_num][rownum])):
            grey[image_num][rownum][colnum] = weightedAverage(image_list[image_num][rownum][colnum])
print("Completed Greyscale Conversion")

plt.imshow(image_resized)
plt.show()
plt.imshow(grey[0], cmap='Greys')
plt.show()
'''
### IMPORT AND RESIZE TESTING
'''
import glob
from scipy import misc
from skimage.transform import resize
import matplotlib.pyplot as plt

path = "/Volumes/TRANSCEND/RPi-Self-Driving-Car/cardataset/Training/Forward/"
image_list = []

#Import and Resize the images
for filename in glob.glob(path+'*.png'):
    image = misc.imread(filename)
    image_resized = resize(image, (18,22), mode='reflect')
    image_list.append(image_resized)
print("Done")

plt.imshow(image)
plt.show()
plt.imshow(image_list[0])
plt.show()
'''

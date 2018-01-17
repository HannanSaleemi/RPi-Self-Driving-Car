from socket import *
import numpy
import numpy as np
from scipy import misc
from skimage.transform import resize

def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]
    print("[* Image Successfully Sent]")

c = socket(AF_INET, SOCK_STREAM)
c.connect(('localhost', 25000))

img = misc.imread('inputimg.png')
print(img)
img = resize(img, (36, 44))

send_from(img, c)

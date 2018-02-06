from socket import *
import numpy
import numpy as np
from scipy import misc
from skimage.transform import resize

def send_img(dest):
    f = open('img2.png', 'rb')
    print("[*] Sending Image")
    l = f.read(150000)
    while (l):
        dest.send(l)
        l = f.read(150000)
    print("[*] Image successfully sent")

c = socket(AF_INET, SOCK_STREAM)
c.connect(('localhost', 25000))

send_img(c)

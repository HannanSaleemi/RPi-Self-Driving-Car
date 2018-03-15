import numpy as np
import cv2
import tkinter
from PIL import Image, ImageTk

#Load color image
img = cv2.imread('red.png')

#Rearrange the color channel from BGR to RGB
b,g,r = cv2.split(img)
img = cv2.merge((r,g,b))

#A root windows to display all the widgets
root = tkinter.Tk()

#Convert the Image object into TkPhoto object
im = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=im)

#Put it in the display windows
tkinter.Label(root, image=imgtk).pack()

#Start the GUI
root.mainloop()

 

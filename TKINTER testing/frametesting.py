from tkinter import Tk, Label, Button, Frame, Entry

root = Tk()
root.title("Training - Model Creation")

##Dataset Configuration
top_frame = Frame(root)

Label(top_frame, text="DATASET CONFIGURATION", font="Helvetica 18 bold").grid(row=0,column=0,columnspan=3)
Label(top_frame, text="Dataset Path").grid(row=1, column=1)

Entry(top_frame, width=20).grid(row=2, column=0, columnspan=2)
Button(top_frame, text="Browse").grid(row=2, column=2)

Label(top_frame, text="Image Size").grid(row=3, column=0)
Entry(top_frame, width=5).grid(row=3, column=1)
Entry(top_frame, width=5).grid(row=3, column=2)

Label(top_frame, text="Model Name").grid(row=4, column=0)
Entry(top_frame, width=20).grid(row=4, column=1, columnspan=2)

top_frame.grid(row=0, column=0, rowspan=5, columnspan=3)
####

##Model Configuraion
bottom_frame = Frame(root)

Label(bottom_frame, text="MODEL CONFIGURATION",  font="Helvetica 18 bold").grid(row=0, column=0, columnspan=4)
Label(bottom_frame, text="Learning Rate").grid(row=1, column=0, columnspan=2, sticky="w")
Label(bottom_frame, text="Epochs").grid(row=2, column=0, sticky="w")
Label(bottom_frame, text="Batch Size").grid(row=3, column=0, columnspan=2, sticky="w")

Entry(bottom_frame, width=20).grid(row=1, column=2, columnspan=2)
Entry(bottom_frame, width=20).grid(row=2, column=2, columnspan=2)
Entry(bottom_frame, width=20).grid(row=3, column=2, columnspan=2)

bottom_frame.grid(row=0, column=4)
####

##Start Training Button
start_frame = Frame(root)

Button(start_frame, text="Start Training").grid(row=0, column=0, pady=20)

start_frame.grid(row=5, column=3)
####

##Image Output
image_frame = Frame(root)

Label(image_frame, text="Training Outputs",  font="Helvetica 18 bold").grid(row=0, column=0, columnspan=2)
Label(image_frame, text="Output Box Here").grid(row=1, column=0)

image_frame.grid(row=7, column=0)
##

##Model Accuracy Statistics
right_frame = Frame(root)

Label(right_frame, text="Model Accuracy Statistics",  font="Helvetica 18 bold").grid(row=0, column=0, columnspan=3)

Label(right_frame, text="Final Reported Accuracy").grid(row=1, column=0, columnspan=2, sticky="w")
Label(right_frame, text="Testing Data Accuracy").grid(row=2, column=0, columnspan=2, sticky="w")

#Should be labels(invisible)
Entry(right_frame, width=15).grid(row=1, column=2, columnspan=2)
Entry(right_frame, width=15).grid(row=2, column=2, columnspan=2)

right_frame.grid(row=7, column=4)
####

root.mainloop()

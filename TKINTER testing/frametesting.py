from tkinter import Tk, Label, Button, Frame, Entry, Text

root = Tk()
root.title("Training - Model Creation")

##Title
Label(root, text="Training - Model Creation", font="Helvetica 18 bold").grid(row=0,column=1, columnspan=3, pady=15)
Button(root, text="Help").grid(row=0, column=4, sticky="e", padx = 15)
####

##Dataset Configuration
data_config_frame = Frame(root)

Label(data_config_frame, text="DATASET CONFIGURATION", font="Helvetica 18 bold").grid(row=0,column=0,columnspan=3)
Label(data_config_frame, text="Dataset Path").grid(row=1, column=1)

Entry(data_config_frame, width=20).grid(row=2, column=0, columnspan=2)
Button(data_config_frame, text="Browse").grid(row=2, column=2)

Label(data_config_frame, text="Image Size").grid(row=3, column=0)
Entry(data_config_frame, width=5).grid(row=3, column=1)
Entry(data_config_frame, width=5).grid(row=3, column=2)

Label(data_config_frame, text="Model Name").grid(row=4, column=0)
Entry(data_config_frame, width=20).grid(row=4, column=1, columnspan=2)

data_config_frame.grid(row=1, column=0, rowspan=5, columnspan=3, padx=10, pady=10)
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

bottom_frame.grid(row=1, column=4, pady=10)
####

##Start Training Button
start_frame = Frame(root)

Button(start_frame, text="Start Training").grid(row=0, column=0, pady=20)

start_frame.grid(row=6, column=3)
####

##Image Output
image_frame = Frame(root)

Label(image_frame, text="Training Outputs",  font="Helvetica 18 bold").grid(row=0, column=0, columnspan=2)
Text(image_frame, height=10, width=40).grid(row=1, column=1)

image_frame.grid(row=8, column=0, padx = 10, pady=20)
####

##Model Accuracy Statistics
right_frame = Frame(root)

Label(right_frame, text="Model Accuracy Statistics",  font="Helvetica 18 bold").grid(row=0, column=0, columnspan=3)

Label(right_frame, text="Final Reported Accuracy").grid(row=1, column=0, columnspan=2, sticky="w")
Label(right_frame, text="Testing Data Accuracy").grid(row=2, column=0, columnspan=2, sticky="w")

#Should be labels(invisible)
Entry(right_frame, width=15).grid(row=1, column=2, columnspan=2)
Entry(right_frame, width=15).grid(row=2, column=2, columnspan=2)

right_frame.grid(row=8, column=4, padx = 10)
####

root.mainloop()

from tkinter import Tk, Label, Button, Frame, Entry

#USE COLUMNSPAN AND ROWSPAN TO AVOID THE MISPLACEMENT OF COLUMNS LIKE THE "X"

root = Tk()
root.title("Training - Model Creation")

Label(root, text="DATASET CONFIGURATION").grid(row=0, column = 1, columnspan=3)
Label(root, text="Dataset Path").grid(row=1, column=1)
dataset_path = Entry(root, width=20).grid(row=2, column=0, columnspan=3)
Button(root, text="Browse").grid(row=2, column=3)

Label(root, text="Image Size", pady=10).grid(row=3, column=0, sticky="w")
Entry(root, width=5).grid(row=3, column=1)
Label(root, text="X").grid(row=3, column=2)
Entry(root, width=5).grid(row=3, column=3)

Label(root, text="Model Name:", padx=10).grid(row=4, column=0)
Entry(root, width=10).grid(row=4, column=1)


Label(root, text="MODEL CONFIGURATION").grid(row=0, column=7, columnspan=1)
Label(root, text="Learning Rate:").grid(row=1, column=7)
Label(root, text="Epochs:").grid(row=2, column=7)
Label(root, text="Batch Size:").grid(row=3, column=7)
Entry(root, width = 10).grid(row=3, column=8)
Entry(root, width = 10).grid(row=2, column=8)
Entry(root, width = 10).grid(row=1, column=8)

Button(root, text="Start Training", padx=10, pady=10).grid(row=5, column = 5, columnspan=2)

Label(root, text="TRAINING OUTPUTS:").grid(row=6, column=5, columnspan=2)
Label(root, text="Training Output:").grid(row=7, column = 0, columnspan=1)


Label(root, text="MODEL ACCURACY STATISTICS:").grid(row=8, column=7, columnspan=2)
Label(root, text="Final Reported Accuracy:").grid(row=9, column=7, columnspan=2)
Label(root, text="Testing Data Accuracy:").grid(row=10, column=7, columnspan=2)
Entry(root, width = 5).grid(row=9, column=9)
Entry(root, width = 5).grid(row=10, column=9)

root.mainloop()




'''
Label(root, text="Old Password").grid(row=0, sticky="w")
Label(root, text="New Password").grid(row=1, sticky="w")
Label(root, text="Enter new password").grid(row=2, sticky="w")

oldpw = Entry(root, width = 16, show='*')
newpw1 = Entry(root, width = 16, show='*')
newpw2 = Entry(root, width = 16, show='*')

oldpw.grid(row=0, column=1, sticky="w")
newpw1.grid(row=1, column=1, sticky="w")
newpw2.grid(row=2, column=1, sticky="w")
'''

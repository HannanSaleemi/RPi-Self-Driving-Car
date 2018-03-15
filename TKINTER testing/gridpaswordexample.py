from tkinter import Tk, Label, Button, Frame, Entry

root = Tk()
root.title("Dataset Configuration")

Label(root, text="Old Password").grid(row=0, sticky="w")
Label(root, text="New Password").grid(row=1, sticky="w")
Label(root, text="Enter new password").grid(row=2, sticky="w")

oldpw = Entry(root, width = 16, show='*')
newpw1 = Entry(root, width = 16, show='*')
newpw2 = Entry(root, width = 16, show='*')

oldpw.grid(row=0, column=1, sticky="w")
newpw1.grid(row=1, column=1, sticky="w")
newpw2.grid(row=2, column=1, sticky="w")

root.mainloop()

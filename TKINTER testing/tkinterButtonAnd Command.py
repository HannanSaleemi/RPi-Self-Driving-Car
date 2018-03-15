from tkinter import Tk, Label, Button

def hello_callback():
    print("Hello")

top = Tk()

#Label
l = Label(top, text = "My Button")
l.pack()

#Button
b = Button(top, text="MyButton", command = hello_callback)
b.pack()

#MainLoop
top.mainloop()

from tkinter import Tk, Label, Button, Frame, Entry

root = Tk()
root.title("Autonomous Vehicle Dashboard")

##Connection Settings
connection_frame = Frame(root)

Label(connection_frame, text="Connection Settings", font="Helvetica 18 bold").grid(row=0,column=0,columnspan=3)
Label(connection_frame, text="Client IP:").grid(row=1, column=0)
Label(connection_frame, text="Client Port:").grid(row=2, column=0)

Entry(connection_frame, width=20).grid(row=1, column=1, columnspan=2)
Entry(connection_frame, width=20).grid(row=2, column=1, columnspan=2)

connection_frame.grid(row=0, column=0, padx=20)
##

##Model Reloading
reloading_frame = Frame(root)

Label(reloading_frame, text="Model Reloading", font="Helvetica 18 bold").grid(row=0,column=0,columnspan=3)
Label(reloading_frame, text="Path to Model:").grid(row=1, column=0, columnspan=3)
Label(reloading_frame, text="STOP Cascade Path:").grid(row=2, column=0, columnspan=3)
Label(reloading_frame, text="Traffic Cascade Path:").grid(row=4, column=0, columnspan=3)

Entry(reloading_frame, width=20).grid(row=1, column=3, columnspan=2)
Entry(reloading_frame, width=20).grid(row=2, column=3, columnspan=2)
Entry(reloading_frame, width=20).grid(row=3, column=3, columnspan=2)

Button()
Button()
Button()

reloading_frame.grid(row=0, column=5)
##

root.mainloop()

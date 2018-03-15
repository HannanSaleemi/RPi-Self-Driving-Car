from tkinter import Label, Button, Frame, Entry

class MyApp(Frame):

    def __init__(self, master=None):

        Frame.__init__(self, master=None)
        self.pack()

        self.grid(column=0, row=0)

        self.l = Label(self, text = "Training - Model Creation")
        self.l.grid(row=0, column=1, padx=15, pady=15)

        self.b = Button(self, text="Hello", command = self.hello)
        self.b.grid(row=1, column=1)

    def hello(self):
            print("Hello")

if __name__ == "__main__":
    MyApp().mainloop()

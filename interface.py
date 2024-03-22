from tkinter import *
from tkinter import ttk


window = Tk()
frame = ttk.Frame(window, padding=10)
frame.grid()

ttk.Label(frame, text="Earsy").grid(column=0, row=0)
ttk.Button(frame, text="Start attendance taking").grid(column=0, row=1)

window.mainloop()
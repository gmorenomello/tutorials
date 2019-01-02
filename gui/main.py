# -*- coding: utf-8 -*-

from tkinter import filedialog,messagebox
from tkinter import *

root = Tk()

def hello():
    print("I still have to implement this functionality!")
    
def open_filename(root):
    """
    Defineds the function that opens the dialog for filetypes
    """
    root.filename =  filedialog.askopenfilename(initialdir = "/",
                                                title = "Select file",
                                                filetypes = (("rmt files","*.jpg"),("hdf5 files","*.h5"),("all files","*.*")))
    print (root.filename)

def open_directory(root):
    """
    Defineds the function that opens the dialog for directory
    """
    root.directory = filedialog.askdirectory()
    print (root.directory)


def save_as_filename(root):
    """
    Defineds the function that opens the dialog for filetypes
    """
    root.filename =  filedialog.asksaveasfilename(initialdir = "/",
                                                  title = "Select file",
                                                  filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (root.filename)

def about():
    messagebox.showinfo("About Yggdrasil", "Yggdrasil was developed by Gustavo Borges Moreno e Mello, 2019")

menubar = Menu(root)

# File handling pull down menu
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open ...", command=lambda: open_filename(root))
filemenu.add_command(label="Open Folder ...", command=lambda: open_directory(root))
filemenu.add_command(label="Save as ...", command=lambda: save_as_filename(root))
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

# Edit pulldown menu
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Cut", command=hello)
editmenu.add_command(label="Copy", command=hello)
editmenu.add_command(label="Paste", command=hello)
menubar.add_cascade(label="Edit", menu=editmenu)

# Help pulldown menu
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)

# display the menu
root.config(menu=menubar)


root.mainloop()
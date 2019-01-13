# -*- coding: utf-8 -*-


#-- Open file
from tkinter import filedialog
from tkinter import *
 
root = Tk()
def open_filename(root):
    """
    Defineds the function that opens the dialog for filetypes
    """
    root.filename =  filedialog.askopenfilename(initialdir = "/",
                                                title = "Select file",
                                                filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print (root.filename)
    
# call the directory dialog box and return the value to a gui property 
Button(root, text="GetFolder", command= lambda:open_filename(root)).pack()
Button(root, text="Quit", command=root.destroy).pack()


#print (root.directory)
root.mainloop()
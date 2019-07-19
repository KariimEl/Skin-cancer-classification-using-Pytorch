import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import torch
from functions import *



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


HEIGHT = 600
WIDTH = 800



def open_file():
    global fileName
    fileName = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    value = tk.StringVar() 
    value.set(fileName)
    entree = tk.Entry(root, textvariable=value,state=tk.DISABLED, width=50)
    entree.grid(row = 3, column = 1, columnspan=3)




def about():
    global help_text
    top = tk.Toplevel()
    top.title("About this application...")
    label = tk.Label(top, text= help_text, justify= tk.LEFT)
    label.pack()
    quit_button= tk.Button(top, text="Fermer", command=top.destroy)
    quit_button.pack()
    

def process():
    global fileName
    model1 = torch.load('model_skin_cancer.pt')
    sortie = skin_cancer(fileName,model1)
    x1 = sortie[0,0]
    x2 = sortie[0,1]

    figure2 = Figure(figsize=(4,3), dpi=100) 
    subplot2 = figure2.add_subplot(111) 
    labels2 = 'Begnin', 'Malegnant'
    pieSizes = [float(x1),float(x2)]
    explode2 = (0, 0.1)  
    subplot2.pie(pieSizes, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True, startangle=90) 
    subplot2.axis('equal')  
    pie2 = FigureCanvasTkAgg(figure2, root) 
    pie2.get_tk_widget().grid(row = 4, column=1, rowspan=7, columnspan=3)


root = tk.Tk()
root.title("Skin Disease Diagnostic")
root.iconbitmap("favicon.ico")
root.geometry("700x600") #Fix the size of the app to be 500x500
root.resizable(0, 0) #Don't allow resizing in the x or y direction

help_text= """The proposed deep learning model is trained and evaluated on the dermoscopic image sets 
from the International Skin Imaging Collaboration (ISIC) 2017 Challenge “Skin Lesion 
Analysis towards Melanoma Detection”. It has been trained 2226 samples and many 
evaluation samples. \n
Designed and programmed by the "Blacklisters - HackBordeaux 2019" \n
Copyright Reserved
"""

intro_text = """      The given results show that the proposed deep learning model achieves 
promising performances on skin lesion segmentation and classification with a percentage 
given after the finish of the process. It doesn’t guarantee that a medical diagnostic is not 
needed, but it gives you an idea about a prospective skin cancer.
"""
   

fileName = "BANNIERE.jpg"

photo = ImageTk.PhotoImage(Image.open(fileName))
W = photo.width()
H = photo.height()
canvas = tk.Canvas(root, width=W, height=H)
canvas.grid(row=1, column = 1,columnspan=3)
canvas.create_image(W//2, H//2, image=photo)

label = tk.Label(root, text= intro_text, justify= tk.LEFT)
label.grid(row=2, column = 1,columnspan=3)

upload_button = tk.Button(root, text="Upload image", command=open_file)
upload_button.grid(row = 3, column = 3)

canvas2 = tk.Canvas(root, width=W, height=H*2)
canvas2.grid(row=4, column = 1,columnspan=3)


about_button = tk.Button(root, text="About", command=about)
about_button.grid(row = 12, column = 1)

process_button = tk.Button(root, text="Proceed to examination", command=process)
process_button.grid(row = 12, column = 2)

quit_button= tk.Button(root, text="Fermer", command=root.destroy)
quit_button.grid(row=12, column = 3)

root.mainloop()






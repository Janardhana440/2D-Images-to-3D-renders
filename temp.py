import _client
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
def ocrf(img_path,Y):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_path)
    output=[]
    for i in result:
        x=i[0][0][0]
        y=i[0][0][1]
        height=i[0][1][0]-x
        width=i[0][3][1]-y
        a=[x,Y-y,height,width,i[1]]
        output.append(a)
    return output

from imutils.perspective import four_point_transform
from imutils import contours
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import*
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import imutils
import numpy as np
import cv2
import os
import PIL
from PIL import ImageTk, Image, ImageDraw
import tkinter as tk
from tkinter import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# for GUI
w = 1000
h = 250
center = h/2
white = (255, 255, 255)
lastx, lasty = None, None
char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 
train_img=[]

# for openCV2
edge = 0
global letters # save each letter
global words # save each word

# input with shape of height=32 and width=128 
inputs = Input(shape=(32,128,1))
 
# convolution layer with kernel size (3,3)
inner = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
inner = MaxPool2D(pool_size=(2, 2), strides=2)(inner)
 
inner = Conv2D(128, (3,3), activation = 'relu', padding='same')(inner)
inner = MaxPool2D(pool_size=(2, 2), strides=2)(inner)
 
inner = Conv2D(256, (3,3), activation = 'relu', padding='same')(inner)
 
inner = Conv2D(256, (3,3), activation = 'relu', padding='same')(inner)
# poolig layer with kernel size (2,1)
inner = MaxPool2D(pool_size=(2, 1))(inner)
 
inner = Conv2D(512, (3,3), activation = 'relu', padding='same')(inner)
# Batch normalization layer
inner = BatchNormalization()(inner)
 
inner = Conv2D(512, (3,3), activation = 'relu', padding='same')(inner)
inner = BatchNormalization()(inner)
inner = MaxPool2D(pool_size=(2, 1))(inner)
 
inner = Conv2D(512, (2,2), activation = 'relu')(inner)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(inner)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
model = tf.keras.Model(inputs, outputs)
model.load_weights('/Users/hans/Desktop/python/sgdo-10000r-30e-7859t-868v.hdf5')

def transformToChar(number):
    if number >= 10 and number < 36:
        return chr(number+55)
    elif number > 35:
        return chr(number+61)
    else:
        return number

def convertImage(img):
    w, h = img.shape

    img = cv2.resize(img, (128,32))
    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255

    plt.imshow(img, cmap='gray')
    plt.show()
    return img


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

def preict():
    words = ""
    letters = []
    train_img=[]
    img = cv2.imread('image.png',cv2.IMREAD_GRAYSCALE)
    img = convertImage(img)
    train_img.append(img)
    train_img = np.asarray(train_img)

    prediction = model.predict(train_img[0:])
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])

    for letter in out:
        for c in letter:
            if int(c) != -1:
                words += char_list[int(c)]
    
    return words


def smoothBrush(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = event.x, event.y


def paint(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), fill='black',width=10,capstyle=ROUND)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=10,joint='curve')
    lastx, lasty = x, y

def convert():
    filename = "image.png"
    image.save(filename)
    pred=preict()
    text.insert(tk.END, pred)


def clear():
    cv.delete('all')
    words = "" # clear words
    letters=[]
    draw.rectangle((0, 0, 1000, 250), fill=(255, 255, 255, 0))
    text.delete('1.0', END)


def save():
    image.save('text_dection_1.png')


win = tk.Tk()
win.geometry("1000x750")
win.configure(bg='black')

frame1 = tk.Frame(win, bg='black')
#frame1.pack(pady=10)
frame1.pack()
#win.resizable(0,0)
cv = Canvas(frame1, bg='white', height=h, width=w)
cv.pack()
# PIL
image = PIL.Image.new("RGB", (w, h), white)
draw = ImageDraw.Draw(image)


txt=tk.Text(win,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
            padx=10,pady=10,height=5,width=20)


cv.bind('<1>', smoothBrush)
cv.pack(expand=YES, fill=BOTH)

frame2 = tk.Frame(win, bg='black')
frame2.pack()
# Text area
text = tk.Text(frame2, width=71, height=5)
text.pack()

# Clear Button
btnClear = Button(frame2, text = "clear", command=clear)
btnClear.pack(padx=20, pady=5)

# Predict Button
btnConvert = Button(frame2, text='convert', command=convert)
btnConvert.pack()

# Save
btnSave = Button(frame2, text='save', command=save)
btnSave.pack()

win.title('Painter')
win.mainloop()

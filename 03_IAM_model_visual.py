import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

text = open('/Users/hans/Desktop/python/handwrting/IAM/words2.txt')
with open('/Users/hans/Desktop/python/handwrting/IAM/words2.txt') as f:
    contents = f.readlines()

images = []
labels = []

train_images = []
train_labels = []
train_input_length = []
train_label_length = []
train_original_text = []

test_images = []
test_labels = []
test_input_length = []
test_label_length = []
test_original_text = []

inputs_length = []
labels_length = []
max_label_len = 0

lines = [line.strip() for line in contents] 
char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara)) 
    return dig_lst



def process_image(img):
    
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    
    img = img.astype('float32')
    
    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255
    
    return img


for index, line in enumerate(lines):
    splits = line.split(' ')
    status = splits[1]
    word_id = splits[0]
    word = "".join(splits[8:])
    lineSplit = line.strip().split(' ')
    fileNameSplit = lineSplit[0].split('-')
    #fileNameSplit = word_id.split('-')
    filepath = '/Users/hans/Desktop/python/handwrting/IAM/words2/'+fileNameSplit[0]+'/'+fileNameSplit[0]+'-'+fileNameSplit[1]+'/'+lineSplit[0]+'.png'
    # process image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    try:
        img = process_image(img)
    except:
        continue
        
    # process label
    try:
        label = encode_to_labels(word)
    except:
        continue
    
    if index % 10 == 0:
        test_images.append(img)
        test_labels.append(label)
        test_input_length.append(31)
        test_label_length.append(len(word))
        test_original_text.append(word)
    else:
        train_images.append(img)
        train_labels.append(label)
        train_input_length.append(31)
        train_label_length.append(len(word))
        train_original_text.append(word)
    
    if len(word) > max_label_len:
        max_label_len = len(word)

train_padded_label = tf.keras.preprocessing.sequence.pad_sequences(train_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

test_padded_label = tf.keras.preprocessing.sequence.pad_sequences(test_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

print(train_padded_label.shape, test_padded_label.shape)


train_images = np.asarray(train_images)
train_input_length = np.asarray(train_input_length)
train_label_length = np.asarray(train_label_length)

test_images = np.asarray(test_images)
test_input_length = np.asarray(test_input_length)
test_label_length = np.asarray(test_label_length)


# input with shape of height=32 and width=128 
input_data = Input(shape=(32,128,1))

# convolution layer with kernel size (3,3)
inner = Conv2D(64, (3,3), activation = 'relu', padding='same')(input_data)
# poolig layer with kernel size (2,2)
inner = MaxPool2D(pool_size=(2, 2), strides=2)(inner)
 
inner = Conv2D(128, (3,3), activation = 'relu', padding='same')(inner)
# poolig layer with kernel size (2,2)
inner = MaxPool2D(pool_size=(2, 2), strides=2)(inner)
 
inner = Conv2D(256, (3,3), activation = 'relu', padding='same')(inner)
inner = Conv2D(256, (3,3), activation = 'relu', padding='same')(inner)
# poolig layer with kernel size (2,1)
inner = MaxPool2D(pool_size=(2, 1))(inner)
 
inner = Conv2D(512, (3,3), activation = 'relu', padding='same')(inner)
# Batch normalization layer
inner = BatchNormalization()(inner)
 
inner = Conv2D(512, (3,3), activation = 'relu', padding='same')(inner)
# Batch normalization layer
inner = BatchNormalization()(inner)
inner = MaxPool2D(pool_size=(2, 1))(inner)
 
inner = Conv2D(512, (2,2), activation = 'relu')(inner)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(inner)
 
# bidirectional LSTM layers with units=128
blstm1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
blstm2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm1)
# include space 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm2)

# model to be used at test time
act_model = Model(input_data, outputs)

the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])

#model to be used at training time
model = Model(inputs=[input_data, the_labels, input_length, label_length], outputs=loss_out)

batch_size = 8
epochs = 30
optimizer_name = 'sgd'

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = optimizer_name, metrics=['accuracy'])

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_IAM01.png')

filepath="{}-{}-{}-{}.hdf5".format(optimizer_name,
                                          str(epochs),
                                          str(train_images.shape[0]),
                                          str(test_images.shape[0]))

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# result = model.fit(x=[train_images, train_padded_label, train_input_length, train_label_length],
#                     y=np.zeros(len(train_images)),
#                     batch_size=batch_size, 
#                     epochs=epochs, 
#                     validation_data=([test_images, test_padded_label, test_input_length, test_label_length], [np.zeros(len(valid_images))]),
#                     verbose=1,
#                     callbacks=callbacks_list)
#load the saved best model weights
# act_model.load_weights('/Users/hans/Desktop/python/sgdo-10000r-30e-7859t-868v.hdf5')

# test = cv2.imread('/Users/hans/Desktop/python/handwrting/IAM/words/a01/a01-000u/a01-000u-05-00.png', cv2.IMREAD_GRAYSCALE)
# test = process_image(test)
# print(test.shape)
# test1=[]
# test1.append(test)
# test1 = np.asarray(test1)
# print(test1[0].shape)
# print(valid_images[0].shape)
# # predict outputs on validation images
# prediction = act_model.predict(test1[0:1])
 
# # use CTC decoder
# out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
#                          greedy=True)[0][0])

# # see the results
# i = 0
# for x in out:
#     print("original_text =  ", valid_original_text[i])
#     print("predicted text = ", end = '')
#     for p in x:  
#         if int(p) != -1:
#             print(char_list[int(p)], end = '')       
#     print('\n')
#     i+=1
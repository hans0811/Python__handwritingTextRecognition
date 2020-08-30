from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

#'balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'
images, label = extract_training_samples('byclass')

X_train, y_train = extract_training_samples('byclass')
X_test, y_test = extract_test_samples('byclass')

print(X_train.shape, y_train.shape,X_test.shape, y_test.shape )
#X_train[0] = np.transpose(X_train[0].values[8,1:].reshape(28, 28), axs=[1, 0])

X_train_reshape = X_train.reshape(len(X_train), 28, 28, 1).astype('float32') 
#X_train_reshape = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_train_normalize = X_train_reshape/255.
print("X_train_normalize_shape: ")
print(X_train_normalize.shape)

X_test_reshape = X_test.reshape(len(X_test), 28, 28, 1).astype('float32') 
#X_test_reshaped  = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_test_normalize = X_test_reshape/255.

categories = 62
FILTER_SIZE = 5
INPUT_SIZE  = 28
MAXPOOL_SIZE = 2

# one_hot to recognize digits and characters
train_onehot = tf.keras.utils.to_categorical(y_train, categories )
test_onehot  = tf.keras.utils.to_categorical(y_test, categories )

model = tf.keras.Sequential()
model.add(keras.layers.Conv2D(filters = 10,
                              kernel_size=(FILTER_SIZE, FILTER_SIZE),
                              padding= 'same',
                              input_shape = (INPUT_SIZE, INPUT_SIZE, 1),
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(keras.layers.Conv2D(filters = 20,
                              kernel_size=(FILTER_SIZE, FILTER_SIZE),
                              padding= 'same',
                              activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dense(units=categories , activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_CHaracters.png')

# log_dir = "logs/fit/" + 'EMNIST'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit(x=X_train_normalize, y=train_onehot, validation_split=0.2, epochs=10, verbose=1, callbacks=[tensorboard_callback])
# scores = model.evaluate(X_test_normalize, test_onehot)
# print(scores)

# model.save('emnist_letter_model_CNN.h5')
# print('Saved model!')
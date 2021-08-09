#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:03:16 2018

@author: ninguem
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from get_image import Board


# 1. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
 
# 2. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)/255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)/255.0

 
# 3. Preprocess class labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



# 4. build the model
model = tf.keras.Sequential()
 
model.add( tf.keras.layers.Input((28,28,1)) )

model.add( tf.keras.layers.Convolution2D(8, (5, 5), padding = 'same', activation='relu') )

model.add( tf.keras.layers.Convolution2D(16, (3, 3), padding = 'same', activation='relu') )

model.add( tf.keras.layers.Flatten() )

model.add( tf.keras.layers.Dense(10, activation='softmax') )




# 5. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 




# 6. Fit model on training data
# model.fit(X_train, y_train, 
#           batch_size=128, epochs=10, verbose=1)
model = tf.keras.models.load_model('trained')



# 7. plot some results...
count = 1
for i in range(5):
    for j in range(5):
        plt.subplot(5,5,count)
        
        x = X_test[count].reshape((1,28,28,1))
        
        y = model.predict(x)
        
        plt.imshow((255*x).reshape(28,28).astype('uint8'))
        plt.title(np.argmax(y))
        plt.axis('off')
 
        count = count+1   
 
plt.show()   
plt.pause(0.001)
 


while True:

    board = Board()
    board.run()
    
    y = np.argmax(model.predict(board.img.reshape(1,28,28,1)))
    
    print(y)
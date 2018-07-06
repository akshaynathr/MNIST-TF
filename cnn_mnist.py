from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# imports 

import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

#Our application logic will be added here

# tf.layers module contains methods to create the following layer types
# 1) conv2d()- Constructs a 2d convolution layer. Takes number of filters,
# filter size, padding and strides as argumet
# 2) max_pooling_2d() - Constructs a 2d pooling layer using max-pooling. 
# Filer size and stride is the argument
# 3) dense()  - Constructs a fully connected layer. Takes number of neurons and activation function as argument.
def _cnn_model_fn(features, labels,mode):
    """ Model function for CNN."""
    # Input layer
    input_layer = tf.reshape(features["x"],[-1,28,28,1])


    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation =tf.nn.relu)

    # Pooling layer
    pool1 = tf.layers.max_pooling2d(inputs= conv1, pool_size=[2,2],strides=2)

    # Convolutional layer #2
    conv2 =tf.layers.conv2d(
        inputs= pool1,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu)
    
    # Pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    # Flatten the last layer for input to Dense layer
    pool2_flatten = tf.reshape(pool2,[-1,7*7*64])

    # Dense layer 
    dense = tf.layers.dense(inputs=pool2_flatten, units= 1024, activation =tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs = dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    
    # Logits layer
    logits = tf.layers.dense(input=dropout,units=10) #linear activation is used.


    #Predictions 

    predictions = {
        # We need to find out 2 things here . 
        # 1) Predict the class (0-9)
        # 2) Predict the probabilities for each class

        "probabilites": tf.nn.softmax(logits , name ="softmax_output"),
        "classes": tf.arg_max(inputs=logits, axis-1)
    }
    

if __name__ == '__main__':
    tf.app.run()
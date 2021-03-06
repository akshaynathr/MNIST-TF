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
    logits = tf.layers.dense(inputs=dropout,units=10) #linear activation is used.


    #Predictions 

    predictions = {
        # We need to find out 2 things here . 
        # 1) Predict the class (0-9)
        # 2) Predict the probabilities for each class

        "probabilites": tf.nn.softmax(logits , name ="softmax_output"),
        "classes": tf.argmax(input=logits, axis=-1)
    }

    # MODE
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    
    # Calculates loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    #Configure the Training Op for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer definition
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    #evaluation metrics
    eval_metrics_op = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions['classes']),

    }


    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metrics_op)


# Main function

def main(arg):
    #Load the data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    # Training data
    train_data = mnist.train.images # Returns np.array
    train_labels =np.asarray(mnist.train.labels,dtype=np.int32)
    # Testing data 
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels,dtype=np.int32)

    # Create estimator

    mnist_classifier = tf.estimator.Estimator(model_fn=_cnn_model_fn,model_dir="/tmp/mnist_conv")


    # Logging Hook

    tensors_to_log = { "probabilities":"softmax_output"}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_fn =tf.estimator.inputs.numpy_input_fn(
                    x={'x':train_data},
                    y=train_labels,
                    batch_size=100,
                    num_epochs=None,
                    shuffle=True)

    
    mnist_classifier.train(input_fn=train_fn,
                        steps=20000,hooks=[logging_hook])
    
    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y= eval_labels,
            num_epochs=1,
            shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print (eval_results)











if __name__ == '__main__':
    tf.app.run(main)
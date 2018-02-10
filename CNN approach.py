import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True
#HELPER

#INIT WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)


#INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

#CONV2D

def conv2d(x,W):
    # x --> input tensor --> [batch,H,W,Channels]
    # W --> [filter H, filter W, Channels IN, Channels OUT]

    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')


#POOLING

def max_pool_2by2(x):
    # x --> input tensor --> [batch,H,W,Channels]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONAL LAYER

def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

# NORMAL (FULLY CONNECTED)

def normal_full_layer(input_layer,size):
    input_size = int(input_layer.shape) 

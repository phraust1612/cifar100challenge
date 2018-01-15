import numpy as np
import tensorflow as tf
import os
from net import Net
from cnn_lenet.init_param import init_lenet

class Lenet (Net):
  def __init__ (self, output:int, load:bool):
    if output != 100 and output != 10:
      raise ValueError ("output must be 10 or 100")
    self.x = tf.placeholder(tf.float32,[None,32,32,3])
    self.y = tf.placeholder(tf.float32,[None,output])
    self.tf_drop = tf.placeholder (tf.float32)
    self.h = 0.025
    self.W = {}
    self.param_dir = "cnn_lenet/param_lenet/"
    self.namelist = os.listdir(self.param_dir)
    if load:
      init_lenet (output)
    self.load ()
    self.build_net ()

  def build_net (self):
    # conv 3x3 + ReLU, + max-pooling
    L = tf.nn.conv2d(self.x, self.W['conv1_0.npy'],strides=[1,1,1,1],padding="SAME")
    L = tf.nn.bias_add (L, self.W['conv1_1.npy'])
    L = tf.nn.relu(L)
    L = tf.nn.max_pool(L,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # conv 3x3 + ReLU, + max-pooling
    L = tf.nn.conv2d(L,self.W['conv2_0.npy'],strides=[1,1,1,1],padding="SAME")
    L = tf.nn.bias_add (L, self.W['conv2_1.npy'])
    L = tf.nn.relu(L)
    L = tf.nn.max_pool(L,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    L = tf.reshape(L,[-1,8*8*64])

    # fc layer + ReLU
    L = tf.matmul(L,self.W['fc1_0.npy']) + self.W['fc1_1.npy']
    L = tf.nn.relu(L)

    # fc layer
    self.output = tf.matmul(L,self.W['fc2_0.npy']) + self.W['fc2_1.npy']

    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.y))
    optimizer = tf.train.AdamOptimizer(learning_rate=self.h)
    self.train = optimizer.minimize(self.loss)

    correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

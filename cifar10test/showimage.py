import sys
import math
import numpy as np
import pickle
import tensorflow as tf
from scipy.misc import toimage

f = open("data/data_batch_1","rb")
dic1 = pickle.load(f,encoding="bytes")
f.close()
f = open("data/data_batch_2","rb")
dic2 = pickle.load(f,encoding="bytes")
f.close()
f = open("data/data_batch_3","rb")
dic3 = pickle.load(f,encoding="bytes")
f.close()
f = open("data/data_batch_4","rb")
dic4 = pickle.load(f,encoding="bytes")
f.close()
f = open("data/data_batch_5","rb")
dic5 = pickle.load(f,encoding="bytes")
f.close()

# Test model and check accuracy
f=open("data/test_batch","rb")
testdic=pickle.load(f,encoding="bytes")
f.close()

#x=tf.placeholder(tf.float32,[None,3072])
x = tf.constant(dic1[b'data'][0],dtype=tf.int32,shape=[3072])
x_img = tf.reshape(x,[3,32,32])
x_img = tf.transpose(x_img,perm=[1,2,0])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    image = x_img.eval()
    print(image.shape)
    print(x)
    print(x_img)
    print(dic1[b'labels'][0])
    xv = sess.run(x)
    print(xv[0],xv[1024],xv[2048])
    xiv = sess.run(x_img)
    print(xiv[0][0])

    toimage(xiv).show()

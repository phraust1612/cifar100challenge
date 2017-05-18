import sys
import math
import numpy as np
import pickle
import tensorflow as tf
from scipy.misc import toimage

# Test model and check accuracy
f=open("data/test_batch","rb")
testdic=pickle.load(f,encoding="bytes")
f.close()

x=tf.placeholder(tf.float32,[None,3072])
x_img = tf.reshape(x,[-1,3,32,32])
x_img = tf.transpose(x_img,perm=[0,2,3,1])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    xiv = sess.run(x_img,feed_dict={x:testdic[b'data']})

    size = len(testdic[b'data'])
    print("total images :",size)
    D = ""
    count=0
    while D != "exit" and count<size:
        toimage(xiv[count]).show()
        D = input("type 'exit' if you want so, or guess it : ")
        if D == str(testdic[b'labels'][count]):
            print("correct!!!")
        else:
            print("wrong!!!",testdic[b'labels'][count])
        count += 1

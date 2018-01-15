import numpy as np
import tensorflow as tf
from resnet import Resnet
from cifar import *

# learning rate decays as scheduled in hlist
#hlist=[]
hlist=[0.0001, 0.0125]
h=0.025
epoch=10000 #default epochs
batch_size=10

print ("setting network...")
net = Resnet()

total_batch = int(data_len/batch_size)
test_batch = int(test_len/batch_size)
last_acc = 0
with tf.Session () as sess:
  sess.run (tf.global_variables_initializer ())
  print ("start training...")
  try:
    for i in range(epoch):
      avg_loss = 0
      avg_acc = 0
      for j in range(total_batch):
        batch_x, batch_y = cifar10train (j*batch_size, (j+1)*batch_size)
        batch_x = np.repeat (batch_x, 7, axis=1)
        batch_x = np.repeat (batch_x, 7, axis=2)
        batch_y = np.eye(10)[batch_y]
        batch_y.transpose()
        feed = {'x':batch_x,'y':batch_y,'drop':0.5}
        c = net.train_param(sess, feed)
        avg_loss += c/batch_size
      print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)

      for j in range(test_batch):
        batch_x, batch_y = cifar10test (j*batch_size, (j+1)*batch_size)
        batch_x = np.repeat (batch_x, 7, axis=1)
        batch_x = np.repeat (batch_x, 7, axis=2)
        batch_y = np.eye(10)[batch_y]
        batch_y.transpose()
        feed = {'x':batch_x,'y':batch_y}
        c = net.get_accuracy (sess, feed)
        avg_acc += c/test_batch

      print('test accuracy:', avg_acc)
  except KeyboardInterrupt:
    print ("stop learning")
  i = input ("save? [y/n] ")
  while i != 'y' and i != 'n':
    i = input ("enter y or n : ")
  if i == 'y':
    print ("saving weights...")
    net.save (sess)

print("learning finished")

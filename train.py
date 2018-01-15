import numpy as np
import tensorflow as tf
from cifar import *
import argparse

# learning rate decays as scheduled in hlist
#hlist=[]
hlist=[0.0001, 0.0125]
h=0.025
epoch=10000 #default epochs
batch_size=10

def start_train (args):
  if args.dataset == "cifar10" or args.dataset == None:
    output = 10
    GetTrainBatch = cifar10train
    GetTestBatch = cifar10test
  else:
    output = 100
    GetTrainBatch = cifar100train
    GetTestBatch = cifar100test

  print ("setting network...")
  if args.net == "Resnet":
    from cnn_resnet.resnet import Resnet
    net = Resnet(output, args.i)
  elif args.net == "Lenet":
    from cnn_lenet.lenet import Lenet
    net = Lenet (output, args.i)
  else:
    return -1

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
          batch_x, batch_y = GetTrainBatch (j*batch_size, (j+1)*batch_size)
          batch_y = np.eye(10)[batch_y]
          batch_y.transpose()
          feed = {'x':batch_x,'y':batch_y,'drop':0.5}
          c = net.train_param(sess, feed)
          avg_loss += c/batch_size
        print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)

        for j in range(test_batch):
          batch_x, batch_y = GetTestBatch (j*batch_size, (j+1)*batch_size)
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

def main ():
  print ("start application...")
  parser = argparse.ArgumentParser ()
  parser.add_argument ("net", help="network name : Resnet / Lenet ")
  parser.add_argument ("--dataset", help="dataset name : cifar10 / cifar100")
  parser.add_argument ("-i", help="to initialize network", action="store_true")
  args = parser.parse_args ()
  if args.net != "Resnet" and args.net != "Lenet":
    print ("Wrong network, choose from Resnet / Lenet")
    return -1
  if args.dataset != None and args.dataset != "cifar100" and args.dataset != "cifar100":
    print ("Wrong dataset, choose from cifar10 / cifar100")
    return -1
  start_train (args)
  return 0

if __name__ == '__main__':
  main ()

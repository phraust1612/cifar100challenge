import scipy.misc
import numpy as np
import pickle
import tensorflow as tf
from resnet import Resnet

# learning rate decays as scheduled in hlist
#hlist=[]
hlist=[0.0001, 0.0125]
h=0.025
epoch=10000	#default epochs
batch_size=10

data = []
label = []
testdata = []
testlabel = []
print ("loading datas...")
for i in range(1,6):
  with open("../data/data_batch_"+str(i),"rb") as f:
    dic = pickle.load(f,encoding="bytes")
    for i in dic[b'data']:
      x = np.array(i,dtype='float32')
      x = x.reshape([3,32,32])
      x = x.transpose([1,2,0])
      x = scipy.misc.imresize (x, (224,224))
      data.append(x)
    for i in dic[b'labels']:
      label.append(i)

# Test model and check accuracy
with open("../data/test_batch","rb") as f:
  dic = pickle.load(f,encoding="bytes")
  for i in dic[b'data']:
    x = np.array(i,dtype='float32')
    x = x.reshape([3,32,32])
    x = x.transpose([1,2,0])
    x = scipy.misc.imresize (x, (224,224))
    testdata.append(x)
  for i in dic[b'labels']:
    testlabel.append(i)

net = Resnet()
print ("loading done...")

total_batch = int(len(data)/batch_size)
test_batch = int(len(testdata)/batch_size)
last_acc = 0
with tf.Session () as sess:
  sess.run (tf.global_variables_initializer ())
  try:
    for i in range(epoch):
	    avg_loss = 0
	    avg_acc = 0
	    for j in range(total_batch):
		    batch_x = np.array(data[j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(label[j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y,'drop':0.5}
		    c = net.train_param(sess, feed)
		    avg_loss += c/batch_size
	    print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)

	    for j in range(test_batch):
		    batch_x = np.array(testdata[j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(testlabel[j*batch_size:(j+1)*batch_size])
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

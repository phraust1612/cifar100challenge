import numpy as np
import pickle
import tensorflow as tf
from resnet import Resnet

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
# this will be used as validation set
f = open("data/data_batch_5","rb")
dic5 = pickle.load(f,encoding="bytes")
f.close()

# Test model and check accuracy
f=open("data/test_batch","rb")
testdic=pickle.load(f,encoding="bytes")
f.close()

# learning rate decays as scheduled in hlist
#hlist=[]
hlist=[0.0001, 0.0125]
h=0.025
epoch=100	#default epochs
batch_size=100

net = Resnet()
total_batch = int(len(dic1[b'data'])/batch_size)
test_batch = int(len(testdic[b'data'])/batch_size)
last_acc = 0
with tf.Session () as sess:
  sess.run (tf.global_variables_initializer ())
  try:
    for i in range(epoch):
	    avg_loss = 0
	    avg_acc = 0
	    val_acc=0
	    for j in range(total_batch):
		    batch_x = np.array(dic1[b'data'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(dic1[b'labels'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y}
		    c = net.train_param(sess, feed)
		    avg_loss += c/batch_size
	    for j in range(total_batch):
		    batch_x = np.array(dic2[b'data'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(dic2[b'labels'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y}
		    c = net.train_param(sess, feed)
		    avg_loss += c/batch_size
	    for j in range(total_batch):
		    batch_x = np.array(dic3[b'data'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(dic3[b'labels'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y}
		    c = net.train_param(sess, feed)
		    avg_loss += c/batch_size
	    for j in range(total_batch):
		    batch_x = np.array(dic4[b'data'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(dic4[b'labels'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y}
		    c = net.train_param(sess, feed)
		    avg_loss += c/batch_size
	    print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)

	    for j in range(total_batch):
		    batch_x = np.array(dic5[b'data'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(dic5[b'labels'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y}
		    c = net.get_accuracy (sess, feed)
		    val_acc += c/batch_size
	    for j in range(test_batch):
		    batch_x = np.array(testdic[b'data'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.array(testdic[b'labels'][j*batch_size:(j+1)*batch_size])
		    batch_y = np.eye(10)[batch_y]
		    batch_y.transpose()
		    feed = {'x':batch_x,'y':batch_y}
		    c = net.get_accuracy (sess, feed)
		    avg_acc += c/total_batch

	    print("val acc:",val_acc,'test accuracy:', avg_acc)
	    if last_acc>val_acc and len(hlist)>0:
		    net.h = hlist.pop()
	    last_acc = val_acc
  except:
    print ("stop learning")
  net.save (sess)

print("learning finished")

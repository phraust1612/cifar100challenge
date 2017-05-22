import sys
import math
import numpy as np
import pickle
import tensorflow as tf

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
default_savefile = "model"

# how to run : python3 (this python file name : arxiv~.py) (model name you set) (number of epochs you want)
if len(sys.argv)>=2:
	default_savefile = sys.argv[1]
if len(sys.argv)>=3:
	epoch = int(sys.argv[2])

x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.float32,[None,10])

# use FC for each layer
# and divide by sqrt(n/2) according to cs231n.stanford.edu
W11 = tf.Variable(tf.random_normal([3072,1024]),name="W11")
W11 = W11 * math.sqrt(2) / (math.sqrt(3) * 1024.0)
b11 = tf.Variable(tf.random_normal([1024]),name="b11")
b11 = b11 * math.sqrt(2) / 32.0
L11 = tf.matmul(x,W11)+b11

W12 = tf.Variable(tf.random_normal([1024,100]),name="W12")
W12 = W12 * math.sqrt(2) / 320.0
b12 = tf.Variable(tf.random_normal([100]),name="b12")
b12 = b12 * math.sqrt(2) / 10.0
L12 = tf.matmul(L11,W12)+b12

W13 = tf.Variable(tf.random_normal([100,10]),name="W13")
W13 = W13 / (math.sqrt(5) * 10.0)
b13 = tf.Variable(tf.random_normal([10]),name="b13")
b13 = b13 / math.sqrt(5)
L13 = tf.matmul(L12,W13)+b13

correct_prediction = tf.equal(tf.argmax(L13, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L13,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=h).minimize(loss)

tf.add_to_collection("vars",W11)
tf.add_to_collection("vars",b11)
tf.add_to_collection("vars",W12)
tf.add_to_collection("vars",b12)
tf.add_to_collection("vars",W13)
tf.add_to_collection("vars",b13)

saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

try:
	new_saver = tf.train.import_meta_graph(default_savefile+".meta")
	new_saver.restore(sess,tf.train.latest_checkpoint("./"))
	allv = tf.get_collection("vars")
except:
	pass

total_batch = int(len(dic1[b'data'])/batch_size)
test_batch = int(len(testdic[b'data'])/batch_size)
last_acc = 0
for i in range(epoch):
	avg_loss = 0
	avg_acc = 0
	val_acc=0
	for j in range(total_batch):
		batch_x = np.array(dic1[b'data'][j*batch_size:(j+1)*batch_size])
		batch_y = np.array(dic1[b'labels'][j*batch_size:(j+1)*batch_size])
		batch_y = np.eye(10)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y}
		c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
		avg_loss += c/batch_size
	for j in range(total_batch):
		batch_x = np.array(dic2[b'data'][j*batch_size:(j+1)*batch_size])
		batch_y = np.array(dic2[b'labels'][j*batch_size:(j+1)*batch_size])
		batch_y = np.eye(10)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y}
		c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
		avg_loss += c/batch_size
	for j in range(total_batch):
		batch_x = np.array(dic3[b'data'][j*batch_size:(j+1)*batch_size])
		batch_y = np.array(dic3[b'labels'][j*batch_size:(j+1)*batch_size])
		batch_y = np.eye(10)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y}
		c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
		avg_loss += c/batch_size
	for j in range(total_batch):
		batch_x = np.array(dic4[b'data'][j*batch_size:(j+1)*batch_size])
		batch_y = np.array(dic4[b'labels'][j*batch_size:(j+1)*batch_size])
		batch_y = np.eye(10)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y}
		c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
		avg_loss += c/batch_size
	saver.save(sess,default_savefile)

	print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)
	for j in range(total_batch):
		batch_x = np.array(dic5[b'data'][j*batch_size:(j+1)*batch_size])
		batch_y = np.array(dic5[b'labels'][j*batch_size:(j+1)*batch_size])
		batch_y = np.eye(10)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y}
		c = sess.run(accuracy,feed_dict=tmpdic)
		val_acc += c/batch_size

	for j in range(test_batch):
		batch_x = np.array(testdic[b'data'][j*batch_size:(j+1)*batch_size])
		batch_y = np.array(testdic[b'labels'][j*batch_size:(j+1)*batch_size])
		batch_y = np.eye(10)[batch_y]
		batch_y.transpose()
		c = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
		avg_acc += c/total_batch
	print("val acc:",val_acc,'test accuracy:', avg_acc)
	if last_acc>val_acc and len(hlist)>0:
		h = hlist.pop()
	last_acc = val_acc
print("learning finished")

import sys
import math
import numpy as np
import pickle
import tensorflow as tf

f = open("../data/train","rb")
dic = pickle.load(f,encoding="bytes")
f.close()

# Test model and check accuracy
f=open("../data/test","rb")
testdic=pickle.load(f,encoding="bytes")
f.close()

# learning rate decays as scheduled in hlist
hlist=[0.0001, 0.0125]
h=0.025
epoch=100	#default epochs
batch_size=100
mix_rate = 0.5
drop_rate = 0.5
default_savefile = "model"

# how to run : python3 (this python file name : arxiv~.py) (model name you set) (number of epochs you want)
if len(sys.argv)>=2:
	default_savefile = sys.argv[1]
if len(sys.argv)>=3:
	epoch = int(sys.argv[2])

x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.float32,[None,100])
tf_drop = tf.placeholder(tf.float32)
x_img = tf.reshape(x,[-1,32,32,3])

# standard conv layer 1
# weight initial state : gaussian distribution with standard deviation 0.5
W1 = tf.Variable(tf.random_normal([3,3,3,192],stddev=0.5),name="W1")
L1 = tf.nn.conv2d(x_img,W1,strides=[1,1,1,1],padding="SAME")
L1 = tf.nn.relu(L1)

# standard conv layer 2
W2 = tf.Variable(tf.random_normal([3,3,192,192],stddev=0.5),name="W2")
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding="SAME")
L2 = tf.nn.relu(L2)

# mlp-conv 1
W3 = tf.Variable(tf.random_normal([1,1,192,96],stddev=0.5),name="W3")
L3 = tf.nn.conv2d(L2, W3,strides=[1,1,1,1],padding="SAME")
L3 = tf.nn.relu(L3)

# mix-pooling layer
L4_max = tf.nn.max_pool(L3,ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME")
L4_avg = tf.nn.avg_pool(L3,ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME")
L4 = mix_rate*L4_max + (1-mix_rate)*L4_avg
L4 = tf.nn.dropout(L4,keep_prob=tf_drop)

# standard conv layer 3
W5 = tf.Variable(tf.random_normal([3,3,96,192],stddev=0.5),name="W5")
L5 = tf.nn.conv2d(L4,W5,strides=[1,1,1,1],padding="SAME")
L5 = tf.nn.relu(L5)

# standard conv layer 4
W6 = tf.Variable(tf.random_normal([3,3,192,192],stddev=0.5),name="W6")
L6 = tf.nn.conv2d(L5,W6,strides=[1,1,1,1],padding="SAME")
L6 = tf.nn.relu(L6)

# mlp-conv layer 2
W7 = tf.Variable(tf.random_normal([1,1,192,192],stddev=0.5),name="W7")
L7 = tf.nn.conv2d(L6,W7,strides=[1,1,1,1],padding="SAME")
L7 = tf.nn.relu(L7)

# mix-pooling layer
L8_max = tf.nn.max_pool(L7,ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME")
L8_avg = tf.nn.avg_pool(L7,ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME")
L8 = mix_rate*L8_max + (1-mix_rate)*L8_avg
L8 = tf.nn.dropout(L8,keep_prob=tf_drop)

# standard conv layer 5
W9 = tf.Variable(tf.random_normal([3,3,192,192],stddev=0.5),name="W9")
L9 = tf.nn.conv2d(L8,W9,strides=[1,1,1,1],padding="SAME")
L9 = tf.nn.relu(L9)

# standard conv layer 6
W10 = tf.Variable(tf.random_normal([3,3,192,192],stddev=0.5),name="W10")
L10 = tf.nn.conv2d(L9,W10,strides=[1,1,1,1],padding="SAME")
L10 = tf.nn.relu(L10)

# mlp-conv layer 3
W11 = tf.Variable(tf.random_normal([1,1,192,192],stddev=0.5),name="W11")
L11 = tf.nn.conv2d(L10,W11,strides=[1,1,1,1],padding="SAME")
L11 = tf.nn.relu(L11)

# mlp-conv layer 4 for reducing dimension
#W12 = tf.Variable(tf.random_normal([1,1,192,64],stddev=0.5),name="W12")
#L12 = tf.nn.conv2d(L11,W12,strides=[1,1,1,1],padding="SAME")
#L12 = tf.nn.relu(L12)
#L12 = tf.reshape(L12,[-1,1024])
L11 = tf.reshape(L11, [-1,3072])

# paper introduced 4th mlp-layer to reduce the dimension, but it showed bad performance
# so I use FC layer instead
# and divide by sqrt(n/2) according to cs231n.stanford.edu
W12 = tf.Variable(tf.random_normal([3072, 1024]),name="W12")
W12 = W12 * math.sqrt(2) / (1024 * math.sqrt(3))
b12 = tf.Variable(tf.random_normal([1024]),name="b12")
b12 = b12 * math.sqrt(2) / 32.0
L12 = tf.matmul(L11,W12)+b12

W13 = tf.Variable(tf.random_normal([1024,100]),name="W13")
W13 = W13 * math.sqrt(2) / 320.0
b13 = tf.Variable(tf.random_normal([100]),name="b13")
b13 = b13 * math.sqrt(2) / 10.0
L13 = tf.matmul(L12,W13)+b13

correct_prediction = tf.equal(tf.argmax(L13, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L13,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=h).minimize(loss)

tf.add_to_collection("vars",W1)
tf.add_to_collection("vars",W2)
tf.add_to_collection("vars",W3)
tf.add_to_collection("vars",W5)
tf.add_to_collection("vars",W6)
tf.add_to_collection("vars",W7)
tf.add_to_collection("vars",W9)
tf.add_to_collection("vars",W10)
tf.add_to_collection("vars",W11)
tf.add_to_collection("vars",W12)
tf.add_to_collection("vars",W13)
tf.add_to_collection("vars",b12)
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

total_batch = int(len(dic[b'data'])/batch_size)
test_batch = int(len(testdic[b'data'])/batch_size)
last_acc = 0
for i in range(epoch):
	avg_loss = 0
	avg_acc = 0
	for j in range(total_batch):
		batch_x = np.array(dic[b'data'][j*100:(j+1)*100])
		batch_y = np.array(dic[b'fine_labels'][j*100:(j+1)*100])
		batch_y = np.eye(100)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y,tf_drop:drop_rate}
		c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
		saver.save(sess,default_savefile)
		avg_loss += c/batch_size
	print("epoch:",str(i+1),"loss=",str(avg_loss))
	for j in range(test_batch):
		batch_x = np.array(testdic[b'data'][j*100:(j+1)*100])
		batch_y = np.array(testdic[b'fine_labels'][j*100:(j+1)*100])
		batch_y = np.eye(100)[batch_y]
		batch_y.transpose()
		c = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y,tf_drop:1})
		avg_acc += c/test_batch
	print('Accuracy:', avg_acc)
	if last_acc>avg_acc and len(hlist)>0:
		h = hlist.pop()
	last_acc = avg_acc
print("learning finished")

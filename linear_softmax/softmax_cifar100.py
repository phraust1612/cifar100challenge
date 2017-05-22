import sys
import numpy as np
import pickle
import math
import tensorflow as tf

hlist=[0.0001,0.0125]
h=0.025
epoch=10
batch_size=100
valrate=0.1
default_savefile = "model"

f = open("../data/train","rb")
tmpdic = pickle.load(f,encoding="bytes")
f.close()

f=open("../data/test","rb")
testdic=pickle.load(f,encoding="bytes")
f.close()

dicsize = int(len(tmpdic[b'data']) * (1.0 - valrate))
dic = {}
dic[b'fine_labels'] = tmpdic[b'fine_labels'][:dicsize]
dic[b'data'] = tmpdic[b'data'][:dicsize]
valdic = {}
valdic[b'fine_labels'] = tmpdic[b'fine_labels'][dicsize:]
valdic[b'data'] = tmpdic[b'data'][dicsize:]

# how to run : python3 (this python file name : arxiv~.py) (model name you set) (number of epochs you want)
if len(sys.argv)>=2:
    default_savefile = sys.argv[1]
if len(sys.argv)>=3:
    epoch = int(sys.argv[2])

x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.float32,[None,100])

W = tf.Variable(tf.random_normal([3072,100]))
W = W * math.sqrt(2) / (math.sqrt(3) * 320.0)
b = tf.Variable(tf.random_normal([100]))
b = b * math.sqrt(2) / 10.0
s = tf.matmul(x,W)+b

tf.add_to_collection("vars",W)
tf.add_to_collection("vars",b)

correct_prediction = tf.equal(tf.argmax(s, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s,labels=y))
optimizer = tf.train.AdagradOptimizer(learning_rate=h).minimize(loss)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if len(sys.argv)>=2:
	new_saver = tf.train.import_meta_graph(sys.argv[1])
	new_saver.restore(sess,tf.train.latest_checkpoint("./"))
	allv = tf.get_collection("vars")
if len(sys.argv)>=3:
	epoch = int(sys.argv[2])

total_batch = int(len(dic[b'data'])/batch_size)
test_batch = int(len(testdic[b'data'])/batch_size)
val_batch = int(len(valdic[b'data'])/batch_size)
last_acc = 0
for i in range(epoch):
    avg_loss = 0
    avg_acc = 0
    val_acc = 0
    for j in range(total_batch):
        batch_x = np.array(dic[b'data'][j*batch_size:(j+1)*batch_size])
        batch_y = np.array(dic[b'fine_labels'][j*batch_size:(j+1)*batch_size])
        batch_y = np.eye(100)[batch_y]
        batch_y.transpose()
        tmpdic = {x:batch_x,y:batch_y}
        c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
        saver.save(sess,default_savefile)
        avg_loss += c/batch_size
    print("epoch:",str(i+1),"loss=",str(avg_loss),"h=",h)

    for j in range(val_batch):
        batch_x = np.array(valdic[b'data'][j*batch_size:(j+1)*batch_size])
        batch_y = np.array(valdic[b'fine_labels'][j*batch_size:(j+1)*batch_size])
        batch_y = np.eye(100)[batch_y]
        batch_y.transpose()
        c = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
        val_acc += c/val_batch

    for j in range(test_batch):
        batch_x = np.array(testdic[b'data'][j*batch_size:(j+1)*batch_size])
        batch_y = np.array(testdic[b'fine_labels'][j*batch_size:(j+1)*batch_size])
        batch_y = np.eye(100)[batch_y]
        batch_y.transpose()
        c = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
        avg_acc += c/test_batch

    print('Test Accuracy:', avg_acc, 'Validation Accuracy:', val_acc)
    if last_acc>val_acc and len(hlist)>0:
        h = hlist.pop()
    last_acc = val_acc

print("learning finished")

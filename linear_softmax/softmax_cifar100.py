import sys
import numpy as np
import pickle
import tensorflow as tf

f = open("../data/train","rb")
dic = pickle.load(f,encoding="bytes")
f.close()

h=0.001
epoch=10
batch_size=100

x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.float32,[None,100])

W6 = tf.Variable(tf.random_normal([3072,100]))
b6 = tf.Variable(tf.random_normal([100]))
L6 = tf.matmul(x,W6)+b6

tf.add_to_collection("vars",W6)
tf.add_to_collection("vars",b6)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L6,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=h).minimize(loss)

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
for i in range(epoch):
	avg_loss = 0
	for j in range(total_batch):
		batch_x = np.array(dic[b'data'][j*100:(j+1)*100])
		batch_y = np.array(dic[b'fine_labels'][j*100:(j+1)*100])
		batch_y = np.eye(100)[batch_y]
		batch_y.transpose()
		tmpdic = {x:batch_x,y:batch_y}
		c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
		saver.save(sess,"linear_model")
		avg_loss += c/batch_size
	print("epoch:",str(i+1),"loss=",str(avg_loss))
print("learning finished")

# Test model and check accuracy
f=open("data/test","rb")
testdic=pickle.load(f,encoding="bytes")
f.close()

total_batch = int(len(testdic[b'data'])/batch_size)
avg_acc = 0
for j in range(total_batch):
	batch_x = np.array(testdic[b'data'][j*100:(j+1)*100])
	batch_y = np.array(testdic[b'fine_labels'][j*100:(j+1)*100])
	batch_y = np.eye(100)[batch_y]
	batch_y.transpose()
	correct_prediction = tf.equal(tf.argmax(L6, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	c = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
	avg_acc += c/total_batch
print('Accuracy:', avg_acc)

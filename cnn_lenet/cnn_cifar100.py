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

h=0.001
epoch=100
batch_size=100
default_savefile = "model"
if len(sys.argv)>=2:
    default_savefile = sys.argv[1]
if len(sys.argv)>=3:
    epoch = int(sys.argv[2])

x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.float32,[None,100])
x_img = tf.reshape(x,[-1,3,32,32])
x_img = tf.transpose(x_img,perm=[0,2,3,1])

W1 = tf.Variable(tf.random_normal([3,3,3,32]),name="W1")
W1 = W1 / (math.sqrt(3) * 12.0)
L1 = tf.nn.conv2d(x_img,W1,strides=[1,1,1,1],padding="SAME")
L1 = tf.nn.relu(L1)

L2 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

W3 = tf.Variable(tf.random_normal([3,3,32,64]),name="W3")
W3 = W3 / 96.0
L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding="SAME")
L3 = tf.nn.relu(L3)

L4 = tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
L4 = tf.reshape(L4,[-1,8*8*64])

W5 = tf.Variable(tf.random_normal([8*8*64, 1024]),name="W5")
b5 = tf.Variable(tf.random_normal([1024]),name="b5")
W5 = W5 * math.sqrt(2) / 2048.0
b5 = b5 * math.sqrt(2) / 32.0
L5 = tf.matmul(L4,W5)+b5
L5 = tf.nn.relu(L5)

W6 = tf.Variable(tf.random_normal([1024,100]),name="W6")
b6 = tf.Variable(tf.random_normal([100]),name="b6")
W6 = W6 * math.sqrt(2) / 320.0
b6 = b6 * math.sqrt(2) / 10.0
L6 = tf.matmul(L5,W6)+b6

tf.add_to_collection("vars",W1)
tf.add_to_collection("vars",W3)
tf.add_to_collection("vars",W5)
tf.add_to_collection("vars",W6)
tf.add_to_collection("vars",b5)
tf.add_to_collection("vars",b6)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L6,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=h).minimize(loss)

correct_prediction = tf.equal(tf.argmax(L6, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

try:
    new_saver = tf.train.import_meta_graph(default_savefile+".meta")
    new_saver.restore(sess,tf.train.latest_checkpoint("./"))
    allv = tf.get_collection("vars")
except:
    pass

total_batch = int(len(dic[b'data'])/batch_size)
test_batch = int(len(testdic[b'data'])/batch_size)

for i in range(epoch):
    avg_loss = 0
    avg_acc = 0
    for j in range(total_batch):
        batch_x = np.array(dic[b'data'][j*100:(j+1)*100])
        batch_y = np.array(dic[b'fine_labels'][j*100:(j+1)*100])
        batch_y = np.eye(100)[batch_y]
        batch_y.transpose()
        tmpdic = {x:batch_x,y:batch_y}
        c,_ = sess.run([loss,optimizer],feed_dict=tmpdic)
        saver.save(sess,default_savefile)
        avg_loss += c/batch_size
    print("epoch:",str(i+1),"loss=",str(avg_loss))

    for j in range(test_batch):
        batch_x = np.array(testdic[b'data'][j*100:(j+1)*100])
        batch_y = np.array(testdic[b'fine_labels'][j*100:(j+1)*100])
        batch_y = np.eye(100)[batch_y]
        batch_y.transpose()
        c = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
        avg_acc += c/test_batch
    print('Test Accuracy:', 100*avg_acc)
print("learning finished")

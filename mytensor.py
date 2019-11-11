from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import cv2

tf.set_random_seed(777)
classes = ['airplane','car','cat','dog','flower','fruit','motorbike','person']
num_classes = len(classes)
path = 'natural_images/'

tmpx = []
tmpy = []
img = None

for idx , class_w in enumerate(classes) :
    label = [0 for i in range(num_classes)]
    label[idx] = 1
    image_dir = path + class_w +'/'
    
    for top, dir, f in os.walk(image_dir) :
        for filename in f :
            #print(image_dir + filename)
            try:
                img = cv2.imread(image_dir+filename)
                #print(img.shape)
                img = img.tolist()
                tmpx.append(img)
                tmpy.append(label)
            except:
                pass
           
tmpx = np.array(tmpx)
tmpy = np.array(tmpy)
print(tmpx)
print(tmpy)
print(tmpx[0].shape)
X_train, X_test, Y_train, Y_test = train_test_split(tmpx,tmpy)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(X_train[0].shape)

learning_rate = 0.01
training_epochs = 10
batch_size = 100

X = tf.placeholder(tf.float32, [None, 30, 30, 3])
Y = tf.placeholder(tf.float32, [None, 8])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') # 30*30*32
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#15*15*32

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')     #15*15*64
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#8*8*64

W3 = tf.Variable(tf.random_normal([8 * 8 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 8 * 8 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([256, 8], stddev=0.01))
hypothesis = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, 
                                                              labels = Y))    #costfunction

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(len(X_train) / batch_size)
    
    for  i in range(total_batch) :
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #batch_xs = batch_xs.reshape(-1,28,28,1)
        feed_dict = {X:X_train, Y:Y_train,keep_prob: 0.7}
        print(X_train.shape)
        cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += cost_val / total_batch
        
    print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost))
    
print('Learning Finished!!')
#
correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={X:X_test.reshape(-1,30,30,3),
                                                   Y:Y_test,keep_prob: 1}))

r = random.randint(0,len(Y_test) -1)
print('Label : ',sess.run(tf.argmax(Y_test[r:r+1],1)))
print('Prediction : ',sess.run(tf.argmax(hypothesis,1),
                               feed_dict={X:X_test[r:r+1].reshape(-1,30,30,3),
                                          keep_prob: 1}))

plt.imshow(X_train[r:r+1].reshape(30,30),cmap = 'Greys',
           interpolation='nearest')
plt.show()
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import random

tf.set_random_seed(777)
test_DIR = 'lego_test/lego/'
test_folder_list = os.listdir(test_DIR)

X = [] # 사진 데이터
Y = [] # 라벨

for index in range(len(test_folder_list)):
    path = os.path.join(test_DIR, test_folder_list[index])
    path += '/'
    img_list = os.listdir(path)
    
    for img in img_list:
        img_path = os.path.join(path, img)
        try:
            img_read = plt.imread(img_path)
            X.append(img_read)
            Y.append(index)
        except:
            pass
        
tmpx = np.array(X)

Y = np.array([[i] for i in Y])
enc = OneHotEncoder()
enc.fit(Y)
tmpy = enc.transform(Y).toarray()

# 전처리 끝
print(tmpx.shape)
print(tmpy.shape)

X_train, X_test, Y_train, Y_test = train_test_split(tmpx,tmpy)
print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)

#이미지 shape 200,200,4
learning_rate = 0.01
training_epochs = 10
batch_size = 100

X = tf.placeholder(tf.float32, [None, 200, 200, 4])
Y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 4, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
#200x200x32
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#100x100x32

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
#100x100x64
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#50x50x64

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev =0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
#50x50x128
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#25x25x128
L3 = tf.nn.dropout(L3, keep_prob)

L3_flat = tf.reshape(L3,[-1, 25 * 25 * 128])
# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([25 * 25 * 128, 4], stddev=0.01))
hypothesis = tf.matmul(L3_flat, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, 
                                                              labels = Y))    #costfunction

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

target1 = tf.nn.softmax(hypothesis)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(len(X_train)/ batch_size)
    
    for  i in range(total_batch) :
        #batch_xs, batch_ys = 
        #batch_xs = batch_xs.reshape(-1,28,28,1)
        feed_dict = {X:X_train, Y:Y_train , keep_prob: 1}
        cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += cost_val / total_batch
        
    print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost))
    
print('Learning Finished!!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
accuracy2 = tf.cast(correct_prediction,tf.float32)
print('Total Accuracy : ', sess.run(accuracy, feed_dict={X:X_test,
                                                   Y:Y_test, keep_prob:1}))

#for i in 

#for i in range() :
#    print('Label'+str(i)+' Accuracy : ', sess.run(accuracy, feed_dict={X:X_test[i],
#                                                   Y:Y_test[i], keep_prob:1}))

r = random.randint(0,4)
print('Label : ',sess.run(tf.argmax(Y_test[r:r+1],1)))
print('Prediction : ',sess.run(tf.argmax(hypothesis,1),
                               feed_dict={X:X_test[r:r+1],
                                          keep_prob: 1}))
print('Prediction : ',sess.run(target1,
                               feed_dict={X:X_test[r:r+1],
                                          keep_prob: 1}) * 100)

plt.imshow(X_test[r:r+1].reshape(200,200,4),cmap = 'Greys',
           interpolation='nearest')
plt.show()
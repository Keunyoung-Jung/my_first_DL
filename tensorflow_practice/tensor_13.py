import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print(mnist)
learning_rate = 0.01
training_epochs = 10
batch_size = 100

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
hypothesis = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, 
                                                              labels = Y))    #costfunction

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(mnist.train._num_examples / batch_size)
    
    for  i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1,28,28,1)
        feed_dict = {X:batch_xs, Y:batch_ys,keep_prob: 0.7}
        cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += cost_val / total_batch
        
    print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost))
    
print('Learning Finished!!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={X:mnist.test.images.reshape(-1,28,28,1),
                                                   Y:mnist.test.labels,keep_prob: 1}))

r = random.randint(0,mnist.test._num_examples -1)
print('Label : ',sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print('Prediction : ',sess.run(tf.argmax(hypothesis,1),
                               feed_dict={X:mnist.test.images[r:r+1].reshape(-1,28,28,1),
                                          keep_prob: 1}))

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap = 'Greys',
           interpolation='nearest')
plt.show()
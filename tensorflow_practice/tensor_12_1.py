import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.1
training_epochs = 15
batch_size = 50

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10])) 

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)    #소프트맥스

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis =1))           #softmax 

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate ).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(mnist.train._num_examples / batch_size)
    
    for  i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}
        cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += cost_val / total_batch
        
    print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost))
    
print('Learning Finished!!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={X:mnist.test.images,
                                                   Y:mnist.test.labels}))

r = random.randint(0,mnist.test._num_examples -1)
print('Label : ',sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print('Prediction : ',sess.run(tf.argmax(hypothesis,1),
                               feed_dict={X:mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap = 'Greys',
           interpolation='nearest')
plt.show()
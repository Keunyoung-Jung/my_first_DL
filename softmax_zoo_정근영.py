import tensorflow as tf
from numpy import *
tf.set_random_seed(777)

xy = loadtxt('data-04-zoo.csv',delimiter = ',',dtype=float32)
x_data = xy[:, :-1]
y_data = xy[:, [-1]]
y_data_pre = zeros((101,7))
print(x_data.shape)

for i in range(len(y_data)) :
    y_data_pre[i][int(y_data[i])] = 1
    
#print(y_data_pre)

X = tf.placeholder(tf.float32, shape =[None,16])
Y = tf.placeholder(tf.float32, shape =[None,7])
nb_classes  = 7

W = tf.Variable(tf.random_normal([16,nb_classes],name = 'weight'))
b = tf.Variable(tf.random_normal([nb_classes],name = 'bias'))
 
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
 
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis =1))
 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
 
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y) ,dtype = tf.float32))
 
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
     
    for step in range(10001) :
        #sess.run(optimizer, feed_dict = {X: x_data, Y: y_data_pre} )
        cost_val, _  = sess.run([cost,optimizer], feed_dict = {X: x_data, Y: y_data_pre})
        if step % 200 == 0 :
            print(step,'\t', cost_val)
             
    print('-------------------------------')
    k = sess.run(hypothesis, feed_dict={X:[[0,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0]]})#5
    print(k, sess.run(tf.argmax(k,1)))
    print('-------------------------------')
    m = sess.run(hypothesis, feed_dict={X:[[0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0]]})#3
    print(m, sess.run(tf.argmax(m,1)))
    print('-------------------------------')
    n = sess.run(hypothesis, feed_dict={X:[[1,0,1,0,1,0,0,0,0,1,1,0,6,0,1,0]]})#5
    print(n, sess.run(tf.argmax(n,1)))
    print('-------------------------------')
    all = sess.run(hypothesis, feed_dict={X:[[0,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0],
                                             [0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0],
                                             [1,0,1,0,1,0,0,0,0,1,1,0,6,0,1,0]]})
    print(all, sess.run(tf.argmax(all,1)))
              
    h,c,a = sess.run([hypothesis,predicted,accuracy],
                     feed_dict={X: x_data, Y: y_data_pre})
    print('\nHypothesis : \n',h,'\nCorrect (Y) : \n',c,'\nAccuracy : ',a)
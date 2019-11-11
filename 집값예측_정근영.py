import tensorflow as tf
from numpy import *
from sklearn.preprocessing import MinMaxScaler

f = open('data.txt')
data = []
sw=0
for i in f :
    j_tmp = []
    if sw == 1 :
        for j in range(len(i.split())) :
            k = float(i.split()[j])
            j_tmp.append(k)
        data.append(j_tmp)
    sw = 1

data = array(data)
scaler = MinMaxScaler()
fitted = scaler.fit(data)
data = scaler.transform(data)
print(data)

xtrain1 = data[:,1]
xtrain2 = data[:,2]
xtrain3 = data[:,3]
xtrain4 = data[:,4]
ytrain = data[:,0]


#===============================================================================
# print(xtrain1)
# print(xtrain2)
# print(xtrain3)
# print(xtrain4)
# print(ytrain)
#===============================================================================

#===============================================================================
# xtrain1 = [1,2,3]
# xtrain2 = [1,2,3]
# xtrain3 = [1,2,3]
# xtrain4 = [1,2,3] 
# ytrain = [1,2,3]
#===============================================================================

W1 = tf.Variable(tf.random_normal([1]), name = 'weight')
W2 = tf.Variable(tf.random_normal([1]), name = 'weight')
W3 = tf.Variable(tf.random_normal([1]), name = 'weight')
W4 = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = xtrain1 * W1 +xtrain2 * W2 +xtrain3 * W3 +xtrain4 * W4 + b
cost = tf.sqrt(tf.reduce_mean(tf.square(hypothesis - ytrain)))
  
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)
  
X1 = tf.placeholder(tf.float32, shape = None)
X2 = tf.placeholder(tf.float32, shape = None)
X3 = tf.placeholder(tf.float32, shape = None)
X4 = tf.placeholder(tf.float32, shape = None)
Y = tf.placeholder(tf.float32, shape = None)
  
init = tf.global_variables_initializer()
with tf.Session() as sess :
    sess.run(init)
    for step in range(10001) :
        cost_val,W_val1,W_val2,W_val3, W_val4, b_val, _ = sess.run([cost,W1,W2,W3,W4,b,train], #cost함수 동작을위해 필요
                                             feed_dict = {X1:xtrain1,X2:xtrain2,X3:xtrain3,X4:xtrain4,Y:ytrain})
          
        if step % 500 == 0 :
            print(cost_val, W_val1,W_val2,W_val3, W_val4, b_val)
              
              
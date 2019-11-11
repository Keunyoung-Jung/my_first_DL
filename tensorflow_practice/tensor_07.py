import tensorflow as tf
from numpy import *

N = 20000
def sigmoid(x):
    return 1/(1+exp(-x))



x_data = random.randn(N,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
wxb = matmul(w_real,x_data.T) + b_real
y_data_pre_noise = sigmoid(wxb)
y_data = random.binomial(1,y_data_pre_noise)

#===============================================================================
# 손실함수
#  y_pred = tf.sigmoid(y_pred)
#  loss = -y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)
#  loss = tf.reduce_mean(loss)
# 
# with tf.name_scope("loss") as scope :
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
#     loss = tf.reduce_mean(loss)
#===============================================================================


import tensorflow as tf

x = tf.constant([[1,2,3],[4,5,6]])
print(x.get_shape())

a = tf.constant([1,2,3])
print(a.get_shape())

a = tf.expand_dims(a,1)
print(a.get_shape())

y = tf.matmul(x,a)
sess = tf.InteractiveSession()
print(y.eval())
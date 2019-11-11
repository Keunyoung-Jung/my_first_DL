import tensorflow as tf
#print(tensorflow.__version__)

a = tf.constant(3)
b = tf.constant(4)
c = tf.pow(a,2)
d = tf.multiply(a, b)
e = tf.subtract(c,d)
f = tf.div(c,d)
g = tf.add(e,f)


sess = tf.Session()

outs = sess.run(g)
sess.close()
print('outs = {}'.format(outs))
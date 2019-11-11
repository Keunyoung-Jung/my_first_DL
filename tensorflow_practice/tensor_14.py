import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) #reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#잘정리 되어있는 이미지들을 받아와서 onehot인코딩을 통해 클래스를 분류해줌
learning_rate = 0.001   #러닝하면서 찾아갈 이동 단위
training_epochs = 10    #훈련을 시킬 횟수
batch_size = 100        #훈련시킬 데이터의 묶음

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X,[-1,28,28,1])  #img 28x28x1(black/white)
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01)) 
#표준편차가 0.01인 3x3매트릭스(깊이1) 데이터를 16개만큼 뽑겠다
#28x28x16
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
#strides를 통해 축을 여러개 만들어 주어 이동방향을 여러방향으로 하게 했으나 가운데 1,1만 적용이 된다 
#제로패딩을통해 사이즈를 동일하게 만듬
L1 = tf.nn.relu(L1)     #relu함수를 이용 음수값을 제거하는 방법으로 사용된다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#컨벌루션된 결과를 max pooling을 통해 최댓값으로 다시 정리 1차원축으로 2칸 2차원축으로도 2칸이동한다
#제로패딩을 통해 크기가 줄지 않아야하지만 2x2필터를 사용하여서 크기가 반으로 줄어든다 depth는 그대로 유지된다(32)
#14x14x16
W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))
#역시 필터를 생성하는데 3x3매트릭스의 깊이가16인 데이터를 32개를 사용한다
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') #14x14x32
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#7x7x32
#===============================================================================
#레이어 추가 부분
W3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev =0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') #7x7x64
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#4x4x64
#===============================================================================
L3_flat = tf.reshape(L3,[-1, 4 * 4 * 64])

W4 = tf.Variable(tf.random_normal([4 * 4 * 64, 10],stddev=0.01))
# 최종 출력값 L2 에서의 출력 7*7*64개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듬
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L3_flat, W4) +b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, 
                                                              labels = Y))
#costfunction
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#adamoptimizer를 사용 adam은 최적값의 근사치일때 넓게 점프해서 한번더 체크해준다
sess = tf.Session()
#세션을 통해 메모리에 설정한것들을 올려준다
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :   #같은 트레이닝데이터를 epoch수만큼 훈련시킨다
    avg_cost = 0                        #오버피팅을 통한 정확도 상승을 목표로 한다
    total_batch = int(mnist.train._num_examples / batch_size)
    #train set에 example들을 batchsize(100)만큼 나누어주어 total batch를 구한다
    for  i in range(total_batch) :      #전체 데이터를 100장씩 훈련시키게 된다
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}    #placeholder로 설정한 그릇에 데이터 담아줌
        cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += cost_val / total_batch      #출력을 위해 avg코스트를 만들어주어 확인
        
    print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost))
    
print('Learning Finished!!')        #15번의 훈련(epoch)을 모두 마치면 출력

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
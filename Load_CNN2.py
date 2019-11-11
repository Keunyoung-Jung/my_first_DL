import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import random
import cv2
import time

def onehot(test_DIR):
    parts_class = []
    test_folder_list = os.listdir(test_DIR)

    X = []
    Y = []
    for index in range(10):#len(test_folder_list)):
        path = os.path.join(test_DIR, test_folder_list[index])
        path += '/'
        img_list = os.listdir(path)
        parts_class.append(test_folder_list[index])
        
        for img in img_list:
            img_path = os.path.join(path, img)
            #print(img)
            try:
                print('#')
                img_read = cv2.imread(img_path)
                #img_read = cv2.resize(img_read , (160,120))
                #cv2.imshow(img,img_read)
                #cv2.waitKey()
                X.append(img_read)
                Y.append(index)
            except:
                pass
        
    tmpx = np.array(X)
    
    Y = np.array([[i] for i in Y])
    enc = OneHotEncoder()
    enc.fit(Y)
    tmpy = enc.transform(Y).toarray()
    print('')
    print(tmpx.shape, tmpy.shape, len(parts_class))
    return tmpx , tmpy , parts_class

st_time = time.time()
dirpath = './class10_low/'
#mkdir_parts(dirpath)
tmpx, tmpy , parts_class = onehot(dirpath)
X_train, X_test, Y_train, Y_test = train_test_split(tmpx,tmpy,random_state = 0)
print(int(st_time))
print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)

#이미지 shape 640,480,3


savefile = 'savedmodel/test_cnn_low_model.ckpt'
#saver = tf.train.import_meta_graph(savefile+'-1000.meta')

with tf.Session() as sess :
    new_saver = tf.train.import_meta_graph(savefile+'-1000.meta')
    new_saver.restore(sess, savefile+'-1000')
    
    tf.get_default_graph()
    
    X = sess.graph.get_tensor_by_name("input:0")
    Y = sess.graph.get_tensor_by_name("output:0")
    keep_prob = sess.graph.get_tensor_by_name("dropout:0")
    hypothesis = sess.graph.get_tensor_by_name("hypothesis:0")
    
    print('model Loaded!!')
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(hypothesis),1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy2 = tf.cast(correct_prediction,tf.float32)
    
    ex_time = time.time() - st_time
    
    print('Excute Time : ', int(ex_time) ,'sec')
    print('Total Accuracy : ', sess.run(accuracy, feed_dict={X:X_test,
                                                       Y:Y_test,keep_prob:1}))
    
    r = random.randint(0,10)
    parts_idx = sess.run(tf.argmax(Y_test[r:r+1],1))[0]
    parts_pre_idx = sess.run(tf.argmax(hypothesis,1),
                             feed_dict={X:X_test[r:r+1],keep_prob:1})[0]
    print('Label : ',sess.run(tf.argmax(Y_test[r:r+1],1)))
    print('Parts_Label :',parts_class[parts_idx])
    #print('Parts_class : ', parts_class)
    print('Prediction : ',sess.run(tf.argmax(hypothesis,1),
                                   feed_dict={X:X_test[r:r+1],keep_prob:1}))
    print('Parts_prediction : ',parts_class[parts_pre_idx])
    
    scores = sess.run(tf.nn.softmax(hypothesis),feed_dict={X:X_test[r:r+1],keep_prob: 1})
    ss = []
    for i in scores[0] :
        aa = i*100
        ss.append(round(aa,2))
    print('Prediction : ',ss)

plt.imshow(X_test[r:r+1].reshape(50,50,3),cmap = 'Greys',
           interpolation='nearest')
plt.show()
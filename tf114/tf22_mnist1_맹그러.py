#pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist   #사용가능/ 자동완성x
from keras.datasets import mnist
import keras 
# print(keras.__version__)  #1.2.2  Using Tensorflow backend. 
import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

##[실습] dnn으로, layer3개 이상으로 맹그러 

# Reshape the input data
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

#1-3 onehotencoding
print(np.unique(y_train))  #[0 1 2 3 4 5 6 7 8 9]
y_train=pd.get_dummies(y_train)
y_train = np.array(y_train)
y_test=pd.get_dummies(y_test)
y_test = np.array(y_test)
print(y_train.shape)   #(60000, 10)




#1. 데이터
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])


#2. 모델구성
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784, 32]), name= 'weight1')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name= 'bias1')       #bias는 w과 동일하게
layer1 = tf.compat.v1.matmul(x, w1)+ b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32, 16]), name= 'weight2') 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name= 'bias2')     
layer2 = tf.nn.softmax(tf.compat.v1.matmul(layer1, w2)+ b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 8]), name= 'weight3')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name= 'bias3')       #bias는 w과 동일하게
layer3 = tf.compat.v1.matmul(layer2, w3)+ b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 10]), name= 'weight4')     
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name= 'bias4')     
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4)+ b4)   #최종 layer = hypothesis (이것 하나로 모델 돌아가게됨 )


#3-1. 컴파일 
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
logits = tf.compat.v1.matmul(layer3, w4)+ b4
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels=y))
# loss= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))  #loss = catagorical_crossentropy

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련
#한줄코드
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss) 


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_v= sess.run([train, loss],
                        feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))
# print(y_predict, y_predict_arg)
y_data_arg = np.argmax(y_test, 1)
# print(y_data_arg)

acc = accuracy_score(y_data_arg, y_predict_arg)
print("acc:" , acc)

sess.close()


# 2000 loss: 0.7005318
# acc: 0.7502
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import time

import tensorflow as tf
tf.compat.v1.set_random_seed(337)  #1.X모드에서 이거 사용



#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

##[실습] yys

# Reshape 
x_train = x_train.reshape(-1, 784).astype("float32")/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

#1-3 onehotencoding
# print(np.unique(y_train))  #[0 1 2 3 4 5 6 7 8 9]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape) #(60000, 784) (60000, 10)
print(x_test.shape, y_test.shape)   #(10000, 784) (10000, 10)
# print(np.unique(y_train))  #[[0. 1.]]


#2. 모델구성
x = tf.compat.v1.placeholder('float', shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#layer구성 
#random_normal 정규균포에서 무작위값 생성/ random_uniform 균등균포에서 무작위값 생성

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784, 128]), name= 'weight1')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]), name= 'bias1')       #bias는 w과 동일하게
layer1 = tf.compat.v1.matmul(x, w1)+ b1
# dropout1 = tf.compat.v1.nn.dropout(layer1, rate = 0.3)

# w2 = tf.compat.v1.get_variable('w2', shape=[128,64],initializer = tf.contrib.layers.xavier_initializer())
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128, 64]), name= 'weight2') 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]), name= 'bias2')     
layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1, w2)+ b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64, 32]), name= 'weight3')    
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name= 'bias3')      
layer3 = tf.nn.selu(tf.compat.v1.matmul(layer2, w3)+ b3)

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32, 10]), name= 'weight4')     
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name= 'bias4')     
hypothesis = tf.compat.v1.matmul(layer3, w4)+ b4    



#3-1. 컴파일 

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= hypothesis, labels=y))  
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


###배치 단위로 훈련 =========================================================================#
#100개씩 600번 훈련 시킨 것이므로 6만번 훈련시킨것임 (작은 배치일 수록 훈련을 많이 시킨것이니까 성능 더 좋아짐)

epochs = 100
batch_size = 100
total_batch = len(x_train)/batch_size    #60000/100 = 600  ###range안에는 float형태x -> int

start_time = time.time()
for step in range(epochs):         
    avg_cost = 0
    for i in range(int(total_batch)):      #100개씩 600번 돌아감
        start = i * batch_size             #0, 100, 200, 300 ...59900
        end = start + batch_size           #100, 200, 300,   ...60000
        # x_train[:100], y_train[:100]

        _, cost_val, w_val, b_val= sess.run([train, loss, w4, b4],
                            # feed_dict={x:x_train[:100], y:y_train[:100]})
                            feed_dict={x:x_train[start:end], y:y_train[start:end]})
        
        avg_cost += cost_val/ total_batch   #loss_v의 모든 값들의 합을 개수로 나눈것 => loss_v의 평균
    print("Epoch:", step + 1, "loss: {:.9f}".format(avg_cost))
            
end_time = time.time()
print("훈련끝")


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))


# print(y_predict, y_predict_arg)
y_data_arg = np.argmax(y_test, 1)
# print(y_data_arg)

acc = accuracy_score(y_data_arg, y_predict_arg)
print("acc:" , acc)
print("tf", tf.__version__, "훈련시간:", end_time - start_time)

sess.close()



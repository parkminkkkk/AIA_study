from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import time

import tensorflow as tf
tf.compat.v1.set_random_seed(337)  #1.X모드에서 이거 사용

import tensorflow as tf
# => tf274gpu 가상환경
# tf.compat.v1.disable_eager_execution() #즉시모드 안해 1.0 False 
# tf.random.set_seed(337)   # 2.X모드에서 이거 사용 

# => tf114gpu 가상환경
tf.compat.v1.set_random_seed(337)  #1.X모드에서 이거 사용
# tf.compat.v1.enable_eager_execution() #즉시모드 해 2.0    True
# print("텐서플로 버전 :", tf.__version__)
# print("즉시실행 모드 :", tf.executing_eagerly())



#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

##[실습] yys

# Reshape 
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

#1-3 onehotencoding
# print(np.unique(y_train))  #[0 1 2 3 4 5 6 7 8 9]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape) #(60000, 784) (60000, 10)
print(x_test.shape, y_test.shape)   #(10000, 784) (10000, 10)
# print(np.unique(y_train))  #[[0. 1.]]

######### cnn모델 ######################################################################################
#2. 모델구성
x = tf.compat.v1.placeholder('float', shape=[None, 28,28,1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#layer구성 
#random_normal 정규균포에서 무작위값 생성/ random_uniform 균등균포에서 무작위값 생성
# model.add(Cov2D(32, kernel_size=(3,3), input_shape=(28,28,1)))  ==>> (26,26,32)로 출력하게 됨 
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 3, 1, 64]), name= 'weight1')     #([3, 3, 1, 32]) :  kernel_size = (3,3), channels, filters(output)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]), name= 'bias1')      #bias는 filtests의 개수와 동일하게 해주면 됨 
layer1 = tf.compat.v1.nn.conv2d(x, w1, strides = [1,1,1,1,], padding = 'SAME')   #  [1,2,2,1] = stride 두칸 전진 할 때 // 양 끝의 1은 shape맞춰주기 위한 숫자일뿐
layer1 += b1
L1_maxpool = tf.nn.max_pool2d(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')  
#(None, 14, 14, 64)

# w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64],initializer = tf.contrib.layers.xavier_initializer())
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,3, 64, 32]), name= 'weight2')    #입력(3,3,64) : 위에 layer에서 주는 값 / 출력(32):임의로 정하는 값  
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name= 'bias2')     
layer2 =tf.compat.v1.nn.conv2d(L1_maxpool, w2, strides = [1,1,1,1,], padding = 'VALID')  
layer2 += b2
L2_maxpool = tf.nn.max_pool2d(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')  
#(None, 6, 6, 32)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 3, 32, 16]), name= 'weight3')    
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name= 'bias3')      
layer3 =tf.compat.v1.nn.conv2d(L2_maxpool, w3, strides = [1,1,1,1,], padding = 'SAME')  
layer3 += b3
#(None, 6,6,16)

#Flatten (2차원 만들어주기)
L_flat = tf.reshape(layer3, [-1, 6*6*16])


#Dense레이어 연결 ===========================================================================================#
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([6*6*16, 100]), name= 'weight4')     
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name= 'bias4')     
layer4 = tf.nn.relu(tf.compat.v1.matmul(L_flat, w4) + b4)    
layer4 = tf.nn.dropout(layer4, rate=0.3)

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 10]), name= 'weight5')     
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name= 'bias5')     
hypothesis = tf.compat.v1.matmul(layer4, w5) + b5
# hypothesis = tf.nn.softmax(hypothesis)



#3-1. 컴파일 
# loss= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))  #loss = catagorical_crossentropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= hypothesis, labels=y))  
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


###배치 단위로 훈련 =========================================================================#
#100개씩 600번 훈련 시킨 것이므로 6만번 훈련시킨것임 (작은 배치일 수록 훈련을 많이 시킨것이니까 성능 더 좋아짐)

epochs = 10
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


# acc: 0.2967
# tf 1.14.0 훈련시간: 475.79706621170044
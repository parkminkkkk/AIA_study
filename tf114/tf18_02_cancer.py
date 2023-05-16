import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#1. 데이터 
x_data, y_data = load_breast_cancer(return_X_y=True)

print(x_data.shape, y_data.shape)   #(569, 30) (569,)
print(y_data[:10])             #[0 0 0 0 0 0 0 0 0 0]


#1-3 onehotencoding
print(np.unique(y_data))  #[0 1]
y_data=pd.get_dummies(y_data)
y_data = np.array(y_data)
print(y_data.shape)   #(569, 2)


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, random_state=337, train_size=0.8, shuffle=True)
print(x_train.shape, y_train.shape)   
print(x_test.shape, y_test.shape)     


#1. 데이터
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
w = tf.Variable(tf.random_normal([30,2]), name = 'weight')
b = tf.Variable(tf.zeros([1,2]), name = 'bias')  #[3]/ [1,3] 통상 모두 가능 

#2. 모델구성
hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(x,w) + b)


#3-1. 컴파일 
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
logits = tf.compat.v1.matmul(x,w) +b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels=y))
# loss= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))  #loss = catagorical_crossentropy

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련
#한줄코드
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_v, w_val, b_val = sess.run([train, loss, w, b ],
                                feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))
# print(y_predict, y_predict_arg)

y_data_arg = np.argmax(y_data, 1)
# print(y_data_arg)

acc = accuracy_score(y_data_arg, y_predict_arg)
print("acc:" , acc)

sess.close()

# 2000 loss: 7.948251
# acc: 0.6274165202108963






import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,2])

#2. 모델구성
# hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) + b)

# model.add(Dense(10, input_shape=2))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 8]), name= 'weight1')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name= 'bias1')       #bias는 w과 동일하게
layer1 = tf.compat.v1.matmul(x, w1)+ b1

# model.add(Dense(7))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 4]), name= 'weight2') 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([4]), name= 'bias2')     
layer2 = tf.nn.sigmoid(tf.compat.v1.matmul(layer1, w2)+ b2)

# model.add(Dense(1, activation='sigmoid'))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 2]), name= 'weight3')     
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([2]), name= 'bias3')     
hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(layer2, w3)+ b3)   #최종 layer = hypothesis (이것 하나로 모델 돌아가게됨 )



#컴파일, 훈련 
# loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))    
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)  

# #4. 평가, 예측
# predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32)) 
# #tf.equal에서 T/F로 반환-> 이후 float로 0/1로 나누고 이를 casting함 / 이후 mean으로 n빵함 => 즉, acc수식 

# with tf.Session() as sess: 
#     sess.run(tf.global_variables_initializer())

#     for step in range(5001):
#         cost_val, _ = sess.run([loss, train], feed_dict = {x:x_data, y:y_data})

#         if step % 200 == 0:
#             print(step, cost_val)

#     h, p, a = sess.run([hypothesis, predicted, accuracy],
#                         feed_dict = {x:x_data, y:y_data})
#     print("예측값: ", h, "\n predicted값:", p, "\n Accuracy:", a)


#  Accuracy: 0.629174


#========================================================================================================#
#3-1. 컴파일 
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
logits = tf.compat.v1.matmul(layer2, w3)+ b3
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels=y))
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
    _, loss_v= sess.run([train, loss],
                        feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))


#3-1. 컴파일 
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
# logits = tf.compat.v1.matmul(layer3, w4)+ b4
# hypothesis = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
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
    _, loss_v= sess.run([train, loss],
                         feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict_arg = (y_predict > 0.5).astype(int).flatten()
# print(y_predict, y_predict_arg)

y_data_arg = y_data.flatten()
# print(y_data_arg)

acc = accuracy_score(y_data_arg, y_predict_arg)
print("acc:" , acc)

# 2000 loss: 0.7236612
# acc: 0.37258347978910367






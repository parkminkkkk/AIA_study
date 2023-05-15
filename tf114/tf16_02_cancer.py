###분류문제** - 2가지만 바꿔주면 됨 ###
#1. hypothesis
 # 한정함수(활성화함수) - sigmoid (0~1사이)
#2. loss = "binary_crossentroy"
 # sigmoid = binary_crossentroy (이진분류) / softmax = categorical_crossentroy (다중분류)

 
import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score


#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

y = y.reshape(-1, 1)  


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=337, train_size=0.8, shuffle=True, stratify=y)
print(x_train.shape, y_train.shape)   #(455, 30) (455, 1)
print(x_test.shape, y_test.shape)     #(114, 30) (114, 1)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)


xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name = 'weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name = 'bias')



#2. 모델 
# hypothesis = tf.compat.v1.matmul(x, w) + b    ##==> 즉, 이 함수를 전체 sigmoid해주면 됨 // sigmoid(x) = 1 / (1 + exp(-x))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp, w) + b)   #sigmoid해주면, 지수승에 x값이 조금만 큰 숫자가 들어가도 0과1에 가까운 수로 값이 몰리게 됨


#3. 컴파일, 훈련 
#3-1. 컴파일
# loss= tf.reduce_mean(tf.square(hypothesis - y))       #mse
loss = tf.reduce_mean(yp*tf.log_sigmoid(hypothesis) + (1-yp)*tf.log_sigmoid(1-hypothesis))    # loss = "binary_crossentroy"

# epsilon = 1e-5
# loss = tf.reduce_mean(yp*tf.log(hypothesis+ epsilon) + (1-yp)*tf.log_sigmoid(1-hypothesis +epsilon))    #  loss의 nan

# loss = "binary_crossentroy"//무조건 반쪽만 돌아감 (왜냐하면, y값이 0일때 뒤쪽만, y값이 1일때는 앞쪽만 살아남아있으므로./.)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)  
train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련



#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    sess.run(tf.compat.v1.global_variables_initializer())
    _, loss_v, w_val, b_val = sess.run([train, loss, w, b],
                                feed_dict={xp:x_train, yp:y_train})
    # print(w_val[0][0])
    if step % 40 == 0:
        print(step, 'loss:', loss_v)
print(type(w_val), type(b_val))

#4. 평가, 예측
y_predict = tf.sigmoid(tf.compat.v1.matmul(xp, w_val) + b_val)    #sigmoid 여기도 씌워줘야함!!!
y_predict = tf.cast(y_predict>0.5, dtype=tf.float32)                  #0.5이상이면 True/False인것을 -> float32를 통해서 0/1로 바꾸겠다// np.round사용해도 됨
y_sess = sess.run(y_predict, feed_dict={xp:x_test})   


print(type(y_sess), type(y_predict))
# print(y_predict)
print(y_sess)

acc = accuracy_score(y_test, y_sess)  
print("acc:" , acc)
mse = mean_squared_error(y_test, y_sess)
print("mse:", mse)

sess.close()


# acc: 0.2894736842105263
# mse: 0.7105263157894737
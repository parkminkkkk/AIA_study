import tensorflow as tf
import numpy as np
tf.set_random_seed(337)


x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]   
y_data = [[0,0,1],     #2
          [0,0,1],
          [0,0,1],
          [0,1,0],     #1
          [0,1,0],
          [0,1,0],
          [1,0,0],     #0
          [1,0,0]]
# (8,4) (8,3)

#1. 데이터
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
w = tf.Variable(tf.random_normal([4,3]), name = 'weight')
b = tf.Variable(tf.zeros([1,3]), name = 'bias')  #[3]/ [1,3] 통상 모두 가능 

#2. 모델구성
hypothesis = tf.matmul(x,w) + b

#3-1. 컴파일 
loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련
#한줄코드
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


# [실습] 맹그러봐요(우선 회귀)~

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_v, w_val, b_val = sess.run([train, loss, w, b ],
                                feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

print(type(w_val), type(b_val))

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

x_test = tf.compat.v1.placeholder(tf.float32, shape= [None,4])

# y_predict = x_test*w_val +b_val # 넘파이랑 텐서1이랑 행렬곱했더니 에러생김, 그래서 밑의 matmul사용하기 
y_predict = tf.compat.v1.matmul(x_test, w_val) + b_val
y_sess = sess.run(y_predict, feed_dict={x_test:x_data})

# print(type(y_sess), type(y_predict))

r2 = r2_score(y_data, y_sess)  
print("r2:" , r2)
mse = mean_squared_error(y_data, y_sess)
print("mse:", mse)

sess.close()

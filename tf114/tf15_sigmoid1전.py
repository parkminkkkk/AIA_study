import tensorflow as tf
tf.compat.v1.set_random_seed(337)

#1. 데이터 
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,3]] #(6,2)
y_data = [[0], [0], [0], [1], [1], [1]]

#######################[실습] sigmoid빼고 우선 만들어보기 ###########################

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))   
y =  tf.compat.v1.placeholder(tf.float32, shape=(None, 1)) 

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1], dtype=tf.float32), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]


#2. 모델 
hypothesis = tf.compat.v1.matmul(x, w) + b


#3. 컴파일, 훈련 
#3-1. 컴파일
loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련


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

x_test = tf.compat.v1.placeholder(tf.float32, shape= [None,2])

y_predict = x_test*w_val +b_val # 넘파이랑 텐서1이랑 행렬곱했더니 에러생김, 그래서 밑의 matmul사용하기 
y_predict = tf.compat.v1.matmul(x_test, w_val) + b_val
y_sess = sess.run(y_predict, feed_dict={x_test:x_data})

# print(type(y_sess), type(y_predict))

r2 = r2_score(y_data, y_sess)  
print("r2:" , r2)
mse = mean_squared_error(y_data, y_sess)
print("mse:", mse)

sess.close()


### TypeError: Expected sequence or array-like, got <class 'tensorflow.python.framework.ops.Tensor'> ###
#y_sess, y_predict : 넘파이형태/ tf형태이므로 에러발생함***

# print(type(y_sess), type(y_predict))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# <class 'numpy.ndarray'> <class 'tensorflow.python.framework.ops.Tensor'>
#==================================================================================================================#


# x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))   #행의 개수는 바뀔 수 있으므로 None으로 명시
# y =  tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
# b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]


# #2. 모델 
# hypothesis = tf.compat.v1.matmul(x, w) + b

# #3. 컴파일, 훈련 
# #3-1. 컴파일
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련

# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     epochs = 2001
#     for step in range(epochs):
#         _, loss_v, w_val, b_val = sess.run([train, loss, w, b],
#                                     feed_dict={x:x_data, y:y_data})
#         if step % 100 == 0:
#             print(step, 'loss:', loss_v, '\n', w_val, b_val)

#     #4. 평가
#     from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

#     y_pred = tf.compat.v1.matmul(x, w_val) + b_val
#     y_predict = sess.run(y_pred, feed_dict={x:x_data})
#     # print(y_predict)
#     print(y_predict)

#     r2 = r2_score(y_data, y_predict)
#     print("r2:" , r2)

#     # rmse = tf.sqrt(mean_squared_error(y_data, y_predict[0]))
#     # rmsetf = sess.run(rmse)
#     # print("rmse:" , rmsetf)

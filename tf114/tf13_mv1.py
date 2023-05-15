###[실습]#############################################

import tensorflow as tf 
tf.compat.v1.set_random_seed(123)

#1.  데이터 
x1_data = [73., 93., 89., 96., 73.]   #국어
x2_data = [80., 88., 91., 98., 66.]   #영어
x3_data = [75., 93., 90., 100., 70.]   #수학
y_data = [152., 185., 180., 196., 142.]   #환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
# x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y =  tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)


#2. 모델 
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)  
train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련


# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# #yys.
# epochs = 2001
# for step in range(epochs):
#     _, loss_v = sess.run([train, loss], #, hypothesis], 
#                                  feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
#     if step %20 == 0:
#         print(step, 'loss:', loss_v) #, '\n', hy_val)
# sess.close()


##################################################################################################################################
#3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# #방법1.
# for step in range(21):
#     _, loss_v, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
#     print(step, loss_v, w1_val, w2_val, w3_val, b_val )
# sess.close()


#방법2.
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     epochs = 2001
#     for step in range(epochs):
#         loss_val, _, w1_val, w2_val, w3_val, b_val = sess.run([loss, train, w1, w2, w3, b], 
#                                                         feed_dict={x1: x1_data,
#                                                          x2: x2_data,
#                                                          x3: x3_data,
#                                                          y: y_data})
#         if step % 100 == 0:
#             print(step, loss_val, w1_val, w2_val, w3_val, b_val)

# #4. 평가
# from sklearn.metrics import r2_score, mean_absolute_error 

# y_predict = x1_data*w1_val + x2_data*w2_val + x3_data*w3_val + b_val
# print(y_predict)

# r2 = r2_score(y_data, y_predict)
# print("r2:" , r2)

# mae = mean_absolute_error(y_data, y_predict)
# print("mae:" , mae)


import tensorflow as tf 
tf.compat.v1.set_random_seed(337)

#1.  데이터 
x_data = [[73, 41, 65],                             #(5,3)
          [92, 98, 11], 
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]                       
y_data = [[152], [185], [180], [205], [142]]        #(5,1)

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))   #행의 개수는 바뀔 수 있으므로 None으로 명시
y =  tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]


#2. 모델 
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b 

### 연산하기 위한 shape ###
# x.shape = (5,3)  y.shaoe = (5,1)   ##hypothesis의 shape는 y데이터의 shape과 같음  
# hy = x*w + b 
#    = (5,3)*w +b                    ##==>>>hy.shape = (5,1) 나와야 함 
# # (5,3)*(?,?) = (5,1)              ##따라서, (3,1)의 weight가 필요함 

#3. 컴파일, 훈련 
#3-1. 컴파일
loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)  
train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        _, loss_v, w_val, b_val = sess.run([train, loss, w, b],
                                    feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, 'loss:', loss_v, '\n', w_val, b_val)

    #4. 평가
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_pred = tf.compat.v1.matmul(x, w_val) + b_val
    y_predict = sess.run([y_pred], feed_dict={x:x_data})
    # print(y_predict)
    print(y_predict[0])

    r2 = r2_score(y_data, y_predict[0])
    print("r2:" , r2)

    rmse = tf.sqrt(mean_squared_error(y_data, y_predict[0]))
    rmsetf = sess.run(rmse)
    print("rmse:" , rmsetf)


'''
r2: 0.5218325827686798
rmse: 15.825453119865994
'''


#=================================================================================#
# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# epochs = 2001
# for step in range(epochs):
#     _, loss_v, w_val, b_val = sess.run([train, loss, w, b],
#                                  feed_dict={x:x_data, y:y_data})
#     if step %20 == 0:
#         print(step, 'loss:', loss_v, '\n', w_val, b_val)
# sess.close()


# #4. 평가
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# y_predict = tf.compat.v1.matmul(x, w_val) + b_val 
# # print(y_predict)
# print(y_predict[0])

# r2 = r2_score(y_data, y_predict[0])
# print("r2:" , r2)

# rmse = tf.sqrt(mean_squared_error(y_data, y_predict[0]))
# rmsetf = sess.run(rmse)
# print("rmse:" , rmsetf)




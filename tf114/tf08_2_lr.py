#실습
#lr수정해서 epoch 100번 이하로 줄이기 
#step = 100이하, w=1.99, b=0.99


import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터 
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)   
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


#2. 모델구성 
hypothesis = x*w +b 

#3-1. 컴파일
loss = tf. reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)

#3-2. 훈련 
with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                             feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})   #train, loss반환하기 위해서는 x,y값 필요함(키,밸류 명시) // train, loss, w, b모두 sess.run을 통해 반환
        if step %20 ==0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)
    #4. 예측
    x_data = [6,7,8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test*w_val + b_val 

    print("[6,7,8] 예측:", sess.run(y_predict, feed_dict={x_test:x_data}))


# tf08_1파일 [6,7,8] 예측: [12.000409 14.000581 16.000753]

# tf08_2파일 
# [6,7,8] 예측: [12.022587 14.032007 16.041428]  **learning_rate=0.08일때**
# step, loss_val, w_val, b_val
# 100 0.000221508 [2.0094204] [-0.03393549]


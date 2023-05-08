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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련 
with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                             feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})   #train, loss반환하기 위해서는 x,y값 필요함(키,밸류 명시) // train, loss, w, b모두 sess.run을 통해 반환
        if step %20 ==0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)




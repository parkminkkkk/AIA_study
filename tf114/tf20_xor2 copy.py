import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

tf.set_random_seed(337)
x_data = np.array([
    [0,0],
    [0,1], 
    [1,0], 
    [1,1], 
], dtype=np.float32)

y_data = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)
##########################################################MAKE IT#########################################

#2. MODEL
# model.add(Desne(10, input_shape=(2,)))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# model.add(Dense(10))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1

# model.add(Dense(7))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 7]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([7]), name='bias2')
layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer1, w2)+ b2)

# model.add(Dense(1, activation='sigmoid'))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([7, 1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias3')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3)+ b3) # 마지막에 sigmoid
# 0 0 0 0 0 0 0 0 0 0
   # 0 0 0 0 0 0 0

# 3-1. COMPILE
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 3-2. PREDICT
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=np.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_data, y:y_data})
    print("예측값:", h, end='\n')
    print("모델값:", p)
    print("ACC: ",  a)
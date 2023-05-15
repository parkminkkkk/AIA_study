import tensorflow as tf
tf.compat.v1.set_random_seed(447)
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1. 데이터 
x, y = fetch_california_housing(return_X_y=True)
print(x.shape, y.shape)   #(20640, 8) (20640,)
print(y[:10])             #[151.  75. 141. 206. 135.  97. 138.  63. 110. 310.]

y = y.reshape(-1, 1)        #(442,1)
# x(442, 10) * w(?,?) = y(442,1)   #weight의 shape : (10,1)// ##텐서1버전의 행렬계산이 가능하도록 y.reshape

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=337, train_size=0.8, shuffle=True)
print(x_train.shape, y_train.shape)   #(353, 10) (353, 1)
print(x_test.shape, y_test.shape)     #(89, 10) (89, 1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), name = 'weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name = 'bias')

#2. 모델
hypothesis = tf.compat.v1.matmul(xp, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-yp))  #mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss)

#3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 201
    for step in range(epochs):
        _, loss_v, w_val, b_val = sess.run([train, loss, w, b],
                                    feed_dict={xp:x_train, yp:y_train})
        if step % 100 == 0:
            print(step, 'loss:', loss_v, '\n', w_val, b_val)

    #4. 평가, 예측 
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_pred = tf.compat.v1.matmul(xp, w_val) + b_val
    y_predict = sess.run([y_pred], feed_dict={xp:x_test})
    # print(y_predict)
    print(y_predict[0])

    r2 = r2_score(y_test, y_predict[0])
    print("r2:" , r2)

    rmse = tf.sqrt(mean_squared_error(y_test, y_predict[0]))
    rmsetf = sess.run(rmse)
    print("rmse:" , rmsetf)



# r2: -5719823.877636985
# rmse: 2766.567458757019
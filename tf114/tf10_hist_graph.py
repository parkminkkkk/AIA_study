import tensorflow as tf
tf.compat.v1.set_random_seed(337)
import warnings 
warnings.filterwarnings('ignore')

#1. 데이터 
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)   
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)


#2. 모델구성 
hypothesis = x*w +b 

#3-1. 컴파일
loss = tf. reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)

#3-2. 훈련 

loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                             feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})   #train, loss반환하기 위해서는 x,y값 필요함(키,밸류 명시) // train, loss, w, b모두 sess.run을 통해 반환
        if step %20 ==0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)
        w_val_list.append(w_val)

    #4. 예측
    x_data = [6,7,8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test*w_val + b_val 

    print("[6,7,8] 예측:", sess.run(y_predict, feed_dict={x_test:x_data}))

# print(loss_val_list)
# print(w_val_list)


#hist 그래프 그리기
import matplotlib.pyplot as plt

# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weight')
# plt.show()

# plt.scatter(w_val_list, loss_val_list)
# plt.xlabel('weight')
# plt.ylabel('loss')
# plt.show()


#hist subplot으로 그리기 
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# loss_val_list
axs[0].plot(loss_val_list)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')

# w_val_list
axs[1].plot(w_val_list)
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('weight')

# w_val_list, loss_val_list
axs[2].scatter(w_val_list, loss_val_list)
axs[2].set_xlabel('weight')
axs[2].set_ylabel('loss')

plt.show()

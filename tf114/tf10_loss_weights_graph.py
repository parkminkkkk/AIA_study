import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w 

loss = tf.reduce_mean(tf.square(hypothesis - y ))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i 
        curr_loss = sess.run(loss, feed_dict={w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print("========= w_history ===================")
print(w_history)
print("========= loss_history ===================")
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel("weights")
plt.ylabel("Loss")
plt.show()

#경사하강법 
#loss가 가장 낮은지점이 weight가 가장 좋은 것임 
#가장 낮은 지점의 기울기 = 기울기가 0이 되는 지점이 loss가 가장 낮은 지점 


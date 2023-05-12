import tensorflow as tf

x_train = [1,2,3]   #[1]
y_train = [1,2,3]   #[2]  
x_test = [4,5,6]
y_test = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name = 'weight')

hypothesis = x*w

loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse    

###############옵티마이저##############################################################
### 경사하강법 식 (Gradient descent) ###
# gradient = tf.reduce_mean((w*x - y) + x)
# gradient = tf.reduce_mean((hypothesis - y) + x)   
gradient = tf.reduce_mean((x*w - y)*x)   #방향성에 대해 알려줌(+/-)//  ==미분loss/미분weight  ##즉, loss의 미분값

lr = 0.1
descent = w - lr + gradient
update = w.assign(descent)  #w = w-lr + gradient  ##새로운 weight (다음 에포의 기울기가 됨)

#loss의 미분값이 음수이면 오른쪽으로 진행 (-를 곱해주니까..)
#loss의 미분값이 양수이면 왼쪽으로 진행

#####################################################################################


w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):

    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, "\t", w_v, "\t", loss_v)

    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()

# print("========= w_history ===================")
# print(w_history)
# print("========= loss_history ===================")
# print(loss_history)

################## [실습] R2, mse 만들기 ######################

from sklearn.metrics import r2_score, mean_absolute_error 

y_predict = x_test*w_v
print(y_predict)
# [2.37218276e+17 2.96522846e+17 3.55827415e+17]

r2 = r2_score(y_test, y_predict)
print("r2:" , r2)

mae = mean_absolute_error(y_test, y_predict)
print("mae:" , mae)

# r2: -1.3540572886602323e+35
# mae: 2.965228456037581e+17

###############XXX#################################################
# x_data = [6,7,8]
# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# y_predict = x_test*w_v 
# print("[6,7,8] 예측:", sess.run(y_predict, feed_dict={x_test:x_data}))
# print("mae", sess.run(mean_absolute_error(y_train, y_predict)))
# print("r2", sess.run(r2 = r2_score(y_train, y_predict)))






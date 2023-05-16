import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(337)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


#[실습] acc 0.25/0.5/0.75맹그러봐 

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))   
y =  tf.compat.v1.placeholder(tf.float32, shape=(None, 1)) 

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1], dtype=tf.float32), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]


#2. 모델 
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)   #sigmoid해주면, 지수승에 x값이 조금만 큰 숫자가 들어가도 0과1에 가까운 수로 값이 몰리게 됨


#3. 컴파일, 훈련 
# logits = tf.compat.v1.matmul(x,w) +b
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels=y))
loss = tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))    
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  

#4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32)) 
#tf.equal에서 T/F로 반환-> 이후 float로 0/1로 나누고 이를 casting함 / 이후 mean으로 n빵함 => 즉, acc수식 

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([loss, train], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x:x_data, y:y_data})
    print("예측값: ", h, "\n predicted값:", p, "\n Accuracy:", a)

'''
예측값:  [[0.51275206]
 [0.7052243 ]
 [0.6129049 ]
 [0.7825895 ]]
 predicted값: [[1.]
 [1.]
 [1.]
 [1.]]
 Accuracy: 0.5
'''


#------------------------------------------------------------------------------------#
# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# epochs = 2001
# for step in range(epochs):
#     _, loss_v, w_val, b_val = sess.run([train, loss, w, b ],
#                                 feed_dict={x:x_data, y:y_data})
#     if step % 20 == 0:
#         print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))

#4. 평가, 예측
# x_test = tf.compat.v1.placeholder(tf.float32, shape= [None,2])
## y_predict = x_test*w_val +b_val # 넘파이랑 텐서1이랑 행렬곱했더니 에러생김, 그래서 밑의 matmul사용하기 
# y_predict = tf.sigmoid(tf.compat.v1.matmul(x_test, w_val) + b_val)    #sigmoid 여기도 씌워줘야함!!!
# y_predict = tf.cast(y_predict>0.5, dtype=tf.float32)                  #0.5이상이면 True/False인것을 -> float32를 통해서 0/1로 바꾸겠다// np.round사용해도 됨
# y_sess = sess.run(y_predict, feed_dict={x_test:x_data})   
# 
# 
## print(type(y_sess), type(y_predict))
## print(y_predict)
## print(y_sess)
# 
# acc = accuracy_score(y_data, y_sess)  
# print("acc:" , acc)
# 
# sess.close()


'''
[[1.]
 [1.]
 [1.]
 [1.]]
acc: 0.5
'''


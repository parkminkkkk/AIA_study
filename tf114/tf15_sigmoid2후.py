###분류문제** - 2가지만 바꿔주면 됨 ###
#1. hypothesis
 # 한정함수(활성화함수) - sigmoid (0~1사이)
#2. loss = "binary_crossentroy"
 # sigmoid = binary_crossentroy (이진분류) / softmax = categorical_crossentroy (다중분류)

 
import tensorflow as tf
tf.compat.v1.set_random_seed(337)

#1. 데이터 
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,3]] #(6,2)
y_data = [[0], [0], [0], [1], [1], [1]]

#######################[실습] sigmoid빼고 우선 만들어보기 ###########################

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))   
y =  tf.compat.v1.placeholder(tf.float32, shape=(None, 1)) 

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1], dtype=tf.float32), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]


#2. 모델 
# hypothesis = tf.compat.v1.matmul(x, w) + b    ##==> 즉, 이 함수를 전체 sigmoid해주면 됨 // sigmoid(x) = 1 / (1 + exp(-x))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)   #sigmoid해주면, 지수승에 x값이 조금만 큰 숫자가 들어가도 0과1에 가까운 수로 값이 몰리게 됨


#3. 컴파일, 훈련 
#3-1. 컴파일
# loss= tf.reduce_mean(tf.square(hypothesis - y))       #mse
loss = tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))    # loss = "binary_crossentroy"
# loss = "binary_crossentroy"//무조건 반쪽만 돌아감 (왜냐하면, y값이 0일때 뒤쪽만, y값이 1일때는 앞쪽만 살아남아있으므로./.)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_v, w_val, b_val = sess.run([train, loss, w, b ],
                                feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

print(type(w_val), type(b_val))

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import numpy as np

x_test = tf.compat.v1.placeholder(tf.float32, shape= [None,2])

# y_predict = x_test*w_val +b_val # 넘파이랑 텐서1이랑 행렬곱했더니 에러생김, 그래서 밑의 matmul사용하기 
y_predict = tf.sigmoid(tf.compat.v1.matmul(x_test, w_val) + b_val)    #sigmoid 여기도 씌워줘야함!!!
y_predict = tf.cast(y_predict>0.5, dtype=tf.float32)                  #0.5이상이면 True/False인것을 -> float32를 통해서 0/1로 바꾸겠다// np.round사용해도 됨
y_sess = sess.run(y_predict, feed_dict={x_test:x_data})   


print(type(y_sess), type(y_predict))
# print(y_predict)
print(y_sess)

acc = accuracy_score(y_data, y_sess)  
print("acc:" , acc)
mse = mean_squared_error(y_data, y_sess)
print("mse:", mse)


sess.close()


#========================================================================================================#


# x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))   #행의 개수는 바뀔 수 있으므로 None으로 명시
# y =  tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name= 'weight')  #weight는 행열연산 해줘야하므로 shape맞춰주기 [x*w = y(hy)] #w의 shape 
# b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name= 'bias')       #bias는 더하기 연산이므로 상관없음 [1]


# #2. 모델 
# # hypothesis = x * w + b
# logits = tf.compat.v1.matmul(x, w) + b
# hypothesis = tf.sigmoid(logits)


# #3. 컴파일, 훈련 
# #3-1. 컴파일
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련

# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     epochs = 2001
#     for step in range(epochs):
#         _, loss_v, w_val, b_val = sess.run([train, loss, w, b],
#                                     feed_dict={x:x_data, y:y_data})
#         if step % 100 == 0:
#             print(step, 'loss:', loss_v, '\n', w_val, b_val)

#     #4. 평가
#     from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

#     y_pred = tf.compat.v1.matmul(x, w_val) + b_val
#     y_predict = sess.run(y_pred, feed_dict={x:x_data})
#     # print(y_predict)
#     print(y_predict)

#     y_pred_cls = np.round(y_pred)
#     acc = np.mean(y_pred_cls == y_data)
#     print("accuracy: ", acc)

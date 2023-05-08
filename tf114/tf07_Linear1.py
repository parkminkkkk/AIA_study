#x,y는 placeholder/ weight, bias는 변수(Variable)

import tensorflow as tf 
tf.set_random_seed(337)

#1. 데이터 
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)  #bias 초기값은 통상 0

#2. 모델구성 
# y = wx + b    
# #텐서플로 연산방식인, 행렬 연산에서 곱하기의 앞뒤가 바뀌면 차이가 발생함 // y = xw +b
#hypothesis:가설
hypothesis = x*w+b     

#3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis-y))   #mse == hypothesis(훈련결과)-y(실제)의 제곱한 결과의 평균 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   #optimizer - 경사하강법 방식으로 옵티마이저 최적화시켜서 적절한 loss값을 찾음 
train = optimizer.minimize(loss)
# model.compile(loss = 'mse', optimizer='sgd') #sgd=통계적 경사하강법// 텐서2의 모델.컴파일과 동일한 코드  


#3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())  #sess명시 후 변수 초기화하고 시작하기!**

epochs = 2001
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))

sess.close()      #자동으로 close되나, 혹시 모르니까 수동 close





























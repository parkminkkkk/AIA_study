import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#1. 데이터 
path = 'd:/study/_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
test_csv= pd.read_csv(path+'test.csv', index_col=0)
x_data = train_csv.drop(['Outcome'], axis=1)
y_data = train_csv['Outcome']

print(x_data.shape, y_data.shape)   #(652, 8) (652,)

#1-3 onehotencoding
print(np.unique(y_data))  #[0 1]
y_data=pd.get_dummies(y_data)
y_data = np.array(y_data)
print(y_data.shape)   #(652, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=123, train_size=0.8, shuffle=True, stratify=y_data
)


#1. 데이터
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,2])

#2. 모델구성
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 8]), name= 'weight1')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name= 'bias1')       #bias는 w과 동일하게
layer1 = tf.compat.v1.matmul(x, w1)+ b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 4]), name= 'weight2') 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([4]), name= 'bias2')     
layer2 = tf.nn.softmax(tf.compat.v1.matmul(layer1, w2)+ b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 8]), name= 'weight3')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name= 'bias3')       #bias는 w과 동일하게
layer3 = tf.compat.v1.matmul(layer2, w3)+ b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 2]), name= 'weight4')     
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([2]), name= 'bias4')     
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4)+ b4)   #최종 layer = hypothesis (이것 하나로 모델 돌아가게됨 )


#3-1. 컴파일 
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
logits = tf.compat.v1.matmul(layer3, w4)+ b4
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels=y))
# loss= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))  #loss = catagorical_crossentropy

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련
#한줄코드
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_v= sess.run([train, loss],
                         feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))
# print(y_predict, y_predict_arg)

y_data_arg = np.argmax(y_data, 1)
# print(y_data_arg)

acc = accuracy_score(y_data_arg, y_predict_arg)
print("acc:" , acc)

sess.close()

# 2000 loss: 1.7906868
# acc: 0.6395705521472392






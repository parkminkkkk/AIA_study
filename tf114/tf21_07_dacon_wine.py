import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. 데이터 
path = 'd:/study/_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]
print(train_csv.shape) #(5497,13)
 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)   #[1 0 1 ... 1 1 1]
print(type(aaa))  #<class 'numpy.ndarray'>
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white'])) #[0 1]


#1-1 결측치 제거 
# print(train_csv.isnull().sum())
# print(train_csv.info())
# train_csv = train_csv.dropna() #결측치없음 

x_data = train_csv.drop(['quality'], axis=1)
print(x_data.shape)                       #(5497, 12)
y_data = train_csv['quality']
print(type(y_data))
print(y_data)
print("y_shape:", y_data.shape)           #(5497,)

#1-2 one-hot-encoding
print('y의 라벨값 :', np.unique(y_data))  #[3 4 5 6 7 8 9]
print(np.unique(y_data, return_counts=True)) # array([  26,  186, 1788, 2416,  924,  152, 5]

import pandas as pd
y_data=pd.get_dummies(y_data)
y_data = np.array(y_data)
print(y_data.shape)                       #(5497, 7)


#1-3 데이터분리 
x_train, x_test, y_train, y_test = train_test_split(
    x_data,y_data, train_size=0.8, random_state=640874)



#1. 데이터
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 12])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,7])


#2. 모델구성
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([12, 8]), name= 'weight1')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name= 'bias1')       #bias는 w과 동일하게
layer1 = tf.compat.v1.matmul(x, w1)+ b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 4]), name= 'weight2') 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([4]), name= 'bias2')     
layer2 = tf.nn.softmax(tf.compat.v1.matmul(layer1, w2)+ b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 8]), name= 'weight3')    # (4,2) (2,a) (a.b) (b, 1) (4,1) 즉,  w의 중간층 layer의 shape에 맞춰주고 처음과 끝에만 x,y의 shape에 맞춰준다
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name= 'bias3')       #bias는 w과 동일하게
layer3 = tf.compat.v1.matmul(layer2, w3)+ b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 7]), name= 'weight4')     
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([7]), name= 'bias4')     
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4)+ b4)   #최종 layer = hypothesis (이것 하나로 모델 돌아가게됨 )



#3-1. 컴파일 
# loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
# logits = tf.compat.v1.matmul(x,w) +b
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels=y))
loss= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))  #loss = catagorical_crossentropy

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
# train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련
#한줄코드
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_v = sess.run([train, loss],
                          feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', loss_v)

# print(type(w_val), type(b_val))


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict_arg = (y_predict > 0.5).astype(int).flatten()
# print(y_predict, y_predict_arg)

y_data_arg = y_data.flatten()
# print(y_data_arg)

acc = accuracy_score(y_data_arg, y_predict_arg)
print("acc:" , acc)

# 2000 loss: 6.2953672
# # acc: 0.737051378674082






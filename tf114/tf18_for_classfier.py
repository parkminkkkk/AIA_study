import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. 데이터 
path = 'd:/study/_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
test_csv= pd.read_csv(path+'test.csv', index_col=0)
dacon_x = train_csv.drop(['Outcome'], axis=1)
dacon_y = train_csv['Outcome']

##
data_list = [load_iris,load_breast_cancer,load_wine, (dacon_x, dacon_y)]

for i in range(len(data_list)):
    if i<3:
        x_data, y_data = data_list[i](return_X_y=True)
    else:
        x_data, y_data = data_list[i]

    print(x_data.shape, y_data.shape)   #(150, 4) (150,)
    #1-3 onehotencoding
    print(np.unique(y_data))  #[0 1 2]
    y_data=pd.get_dummies(y_data)
    y_data = np.array(y_data)
    print(y_data.shape)   #(150, 3)


    x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, random_state=337, train_size=0.8, shuffle=True)
    print(x_train.shape, y_train.shape)   
    print(x_test.shape, y_test.shape)     


    #1. 데이터
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
    y = tf.compat.v1.placeholder(tf.float32, shape=[None,y_data.shape[1]])
    w = tf.Variable(tf.random_normal([x_data.shape[1],y_data.shape[1]]), name = 'weight')
    b = tf.Variable(tf.zeros([1,y_data.shape[1]]), name = 'bias')  #[3]/ [1,3] 통상 모두 가능 

    #2. 모델구성
    hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) + b)


    #3-1. 컴파일 
    # loss= tf.reduce_mean(tf.square(hypothesis - y))  #mse
    logits = tf.compat.v1.matmul(x,w) +b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= logits, labels=y))
    # loss= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis = 1))  #loss = catagorical_crossentropy

    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)  
    # train = optimizer.minimize(loss)  #loss를 최소화하는 방향으로 훈련
    #한줄코드
    train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss) 


    #3-2. 훈련
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        _, loss_v, w_val, b_val = sess.run([train, loss, w, b ],
                                    feed_dict={x:x_data, y:y_data})
        if step % 10 == 0:
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

    # Print the data name and accuracy
    if i < 3:
        print("Data Name:", data_list[i].__name__)
    else:
        print("Data Name: dacon_diabetes")
    print("Accuracy:", acc)
    
    sess.close()









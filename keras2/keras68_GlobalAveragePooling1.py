# GlobalAveagePooling 
# 전체 공간 범위에서 각 채널의 평균값을 취하여 CNN의 기능 맵의 공간 차원을 줄이는 기술

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np
import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


print(np.unique(y_train,return_counts=True)) 
#np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#one-hot-coding
print(y_train)       #[5 0 4 ... 5 6 8]
print(y_train.shape) #(60000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)       #[[0. 0. 0. ... 0. 0. 0.]..[0.0.0]]
print(y_train.shape) #(60000, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,784)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#2. 모델구성 
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(28,28,1))) 
model.add(MaxPooling2D()) #(2,2) 중 가장 큰 값 뽑아서 반의 크기(14x14)로 재구성함 
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid', activation='relu')) 
model.add(Conv2D(33, 2))  #kernel_size=(2,2)/ (2,2)/ (2) 동일함 
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax')) #np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]-> output_dim에 '10'

model.summary()

#CNN : Conv2D의 연산량이 많음/ 그러나, Flatten한 이후 연산량이 더 많음(쭉 펴서 값을 보기 위함인데, 연산량이 많음) 
# => 이렇게 까지 많은 연산량을 할 필요가 있는가?
###==> GlobalAveragePooling // 연산량 더 적음 ###
# 각 필터별로 평균을 냄 -> 33개(filters의 수)의 값이 나오게 됨 / 대표값들로 평균을 낸 것들 중에서 값을 내자 
# 필터값의 숫자 만큼의 노드가 생긴다  (필터의 개수만큼 노드의 개수를 만들어준다)

'''
##################  Flatten의 연산량 #############################
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 32)        4128
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0
_________________________________________________________________
dense (Dense)                (None, 10)                46090
=================================================================
Total params: 58,762
Trainable params: 58,762
Non-trainable params: 0
_________________________________________________________________

############## Global_average_pooling2d의 연산량 #################
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 32)        4128
_________________________________________________________________
global_average_pooling2d (Gl (None, 32)                0            #filters만큼 (None, 32) // Pooling은 연산량 없음 
_________________________________________________________________
dense (Dense)                (None, 10)                330
=================================================================
Total params: 13,002
Trainable params: 13,002
Non-trainable params: 0
_________________________________________________________________

'''


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=30, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

import time
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, 
          callbacks=(es))
end = time.time()
print("걸린시간:", end-start)


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss:', results[0]) #loss, metrics(acc)
print('acc:', results[1]) #loss, metrics(acc)

# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
# y_test = np.argmax(y_test, axis=1)
# acc = accuracy_score(y_test, y_pred)
# print('pred_acc:', acc)

#=====================================================================================#
# Flatten 
# 걸린시간: 73.55379271507263
# loss: 0.07692945748567581
# acc: 0.9850000143051147

# Global_average_pooling2d
# 걸린시간: 75.32713007926941
# loss: 0.3081895112991333
# acc: 0.9046000242233276











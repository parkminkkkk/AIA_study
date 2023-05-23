#Tensorboard : ensorFlow 모델의 훈련 과정을 시각화하기 위한 도구
# 콜백 함수를 사용하여 모델을 훈련할 때 모델의 성능 지표, 가중치 및 편향 분포, 계층 구조 등 다양한 정보를 TensorBoard를 통해 시각적으로 확인 가능 


from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
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


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                        factor=0.8)
tb = TensorBoard(log_dir='c:/AIA_study/AIA_study/_save/_tensorboard/_graph', 
                 histogram_freq = 0,    # 히스토그램을 기록할 빈도를 설정 // 0: 기록x / 1: 매 epoch마다 기록
                 write_graph=True,      # 모델의 그래프 구조를 저장할지 여부 // 디폴트: True(저장)
                 write_images=True)     # 그래프 시각화를 위해 이미지를 저장할지 여부

# TensorBoard 실행방법
# cmd창 : C:\AIA_study\AIA_study\_save\_tensorboard>cd _graph 경로까지 이동
# tensorboard --logdir=.
# 웹창  : http://localhost:6006/  
# 로컬번호 :  127.0.0.1:6006

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, 
                 callbacks=[es, rlr, tb])
end = time.time()
print("걸린시간:", end-start)


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss:', results[0]) #loss, metrics(acc)
print('acc:', results[1]) #loss, metrics(acc)

model.save('./_save/keras70_1_mnist_graph.h5')

print(hist)  
# <tensorflow.python.keras.callbacks.History object at 0x000002A49542B2B0>
print(hist.history)  #딕셔너리가 모여있는 리스트 형태 
# {'loss': [1.710621953010559, 1.1540658473968506, 0.9113472700119019, 0.8230547904968262, 0.781936526298523], 
#  'acc': [0.36918750405311584, 0.6115000247955322, 0.6913750171661377, 0.7171249985694885, 0.729812502861023], 
#  'val_loss': [1.299047827720642, 0.9557056427001953, 0.8391696810722351, 0.7774842381477356, 0.7519260048866272], 
#  'val_acc': [0.5555833578109741, 0.687250018119812, 0.7149166464805603, 0.7385833263397217, 0.737500011920929]}

import joblib
joblib.dump(hist.history,'./_save/keras70_1_history.dat')   #hist저장, 경로 
#데이터 안에 딕셔너리 형태로 저장되어있음(hist.history) : hist.history 안에 키 밸류 형태로 loss,acc들어가 있음



###################### 시각화 ###############################################################
import matplotlib.pyplot as plt 
plt.figure(figsize=(9,5))

#1. 
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2. 
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label = 'acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()




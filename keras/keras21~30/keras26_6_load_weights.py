#load_weights

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets['target']
# print(type(x)) #<class 'numpy.ndarray'>
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=1234)

#data scaling(스케일링)
scaler = MinMaxScaler() 
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train) #위에 두줄 한줄로 합침 fit_transform
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test)) 


#2. 모델구성 (함수형모델) 
input1 = Input(shape=(13,), name='h1') 
dense1 = Dense(30, name='h2')(input1) 
dense2 = Dense(20, name='h3', activation='relu')(dense1)                                                            
dense3 = Dense(10, name='h4',activation='relu')(dense2)     
output1 = Dense(1, name='h5',)(dense3)
model = Model(inputs=input1, outputs=output1) 
#weights는 모델명시 안되어있으므로 모델구성해줘야함 (weights만 가져옴)


model.load_weights('./_save/keras26_5_save_weights1.h5')
# RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
# => 모델구성 다음에 save한것이었기 때문에 가중치 저장이 되어있지 않아서, compile,loss해야함 
# 초기 랜덤값의 weight만 저장되어 있다. 

model.load_weights('./_save/keras26_5_save_weights2.h5')
# RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
# => load_weights는 compile값 넣어줘야함! 그래서 통상 model.save씀 
# 왜냐하면, weights(가중치)만 저장되어있고 / 컴파일할때 지표'mse'같은건 저장안되었으므로..같은 지표로 compile해줘야함 


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=30) 

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)









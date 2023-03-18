# 모델구성에서 model.save

#input_shape 사용 
#함수형 모델구성(Model), input_layer 따로 명시해줘야함(Input)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
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
    x,y, train_size=0.8, random_state=1234,)

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

model.summary()
'''
Model: "model"
input_1 (InputLayer)   [(None, 13)]  0 #함수형 모델에는 inputlayer있음 
'''

#모델 저장
model.save('./_save/keras26_1_save_model.h5')



'''
#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
'''



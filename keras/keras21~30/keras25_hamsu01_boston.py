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
scaler.fit(x_train) #x_train범위만큼 잡아라
x_train = scaler.transform(x_train) #변환
#x_train의 변환 범위에 맞춰서 하라는 뜻이므로 scaler.fit할 필요x 
x_test = scaler.transform(x_test) #x_train의 범위만큼 잡아서 변환하라 

print(np.min(x_test), np.max(x_test)) 
#-0.0019120458891013214 1.1629424392720045
#범위 밖 데이터이므로, 성능이 더 좋아질 수 있다. 


#2. 모델구성

#함수형과 시퀀스의 차이# 


model = Sequential()
model.add(Dense(30, input_shape=(13, ), name='S1'))       #스칼라13개, 벡터1개 : 열의 개수를 벡터 형식으로 표시 #열이 13개 있다는 뜻 
model.add(Dense(20, name='S2'))
model.add(Dense(10, name='S3'))
model.add(Dense(1, name='S4'))
model.summary()
'''
Model: "sequential"
'''

input1 = Input(shape=(13,), name='h1')                     #name='aaa' : 이름지어줄 수 있음
dense1 = Dense(30, name='h2')(input1)                      #dense1은 input1에 달라붙을거야    
dense2 = Dense(20, name='h3', activation='relu')(dense1)   #dense2는 dense1에 달라붙을거야                                                       
dense3 = Dense(10, name='h4',activation='relu')(dense2)    #전의 layer가 꽁다리에 달라붙음 
output1 = Dense(1, name='h5',)(dense3)
model = Model(inputs=input1, outputs=output1)              #시작과 끝을 명시해줘야 함 (시작, 끝) #함수명 끝에 명시 

model.summary()

'''
Model: "model"
input_1 (InputLayer)   [(None, 13)]  0 #함수형 모델에는 inputlayer있음 
'''


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)



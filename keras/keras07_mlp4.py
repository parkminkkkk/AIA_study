#x는 3개, y는 1개
#X : range(n)= 0~n-1까지 : 시작숫자 0부터, 끝은 n-1 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
'''
print(x)
#range(n)= 0~n-1까지 : 시작숫자 0부터, 끝은 n-1 
[[  0  21 201]
 [  1  22 202]
 [  2  23 203]
 [  3  24 204]
 [  4  25 205]
 [  5  26 206]
 [  6  27 207]
 [  7  28 208]
 [  8  29 209]
 [  9  30 210]]
'''

print(x.shape) #(3, 10) : 3행 10열
x = x.T #(10,3)

y= np.array([[1,2,3,4,5,6,7,8,9,10]]) 
print(y.shape) # (1,10)
y = y.T # (10.1) 

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=3)) #x값(10,3) input_dim 열의 개수에 맞춰야함 -> '3'
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(33))
model.add(Dense(30))
model.add(Dense(1)) # y값(10,1) output_dim 열의 개수에 맞춰야함 -> '1'
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=100, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print("loss :", loss)

result = model. predict([[9, 30, 210]])
print("[9, 30, 210]의 예측값 : ", result)

#[9, 30, 210]의 예측값 : [[10.000014]] : Dense(10,35,20,17,9,1), mse, epochs=500, batch_size=2
#[9, 30, 210]의 예측값 : [[9.999889]] : Dense(10,35,20,17,9,1), mse, epochs=1000, batch_size=2
#[9, 30, 210]의 예측값 : [[9.766971]] : Dense(10,30,20,10,5,1), mse, epochs=100, batch_size=1
#[9, 30, 210]의 예측값 : [[6.540866]] : Dense(10,30,20,10,5,1), mae, epochs=100, batch_size=1
#[9, 30, 210]의 예측값 : [[10.000009]] : Dense(10,20,50,35,7,1), mse, epochs=1000, batch_size=2



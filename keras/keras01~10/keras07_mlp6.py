# x는 3개 y는 3개 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]) #(3,10)
x = x.T #(10,3)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]]) # (3,10)
y = y.T # (10.3) 

# Q.[실습] 예측 [[9, 30, 210]] -> 예상 y값 [[10, 1.9, 0]]

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3)) #x값(10,3) input_dim 열의 개수에 맞춰야함 -> '3'
model.add(Dense(10))
model.add(Dense(37))
model.add(Dense(45))
model.add(Dense(100))
model.add(Dense(3)) # y값(10,3) output_dim 열의 개수에 맞춰야함 -> '3'
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print("loss :", loss)

result = model. predict([[9, 30, 210]])
print("[9, 30, 210]의 예측값 : ", result)

'''
[9, 30, 210]의 예측값 / 예상 y값 [10, 1.9, 0]
[[ 9.996512    1.8663608  -0.01158637]] : Dense(15,70,10,60,20,3) ,mse, epochs=100, batch_size=1
[[10.393967    1.1488692   0.07469627]] : Dense(10,70,100,60,20,3) ,mse, epochs=500, batch_size=1
[[10.05175     1.9005717  -0.10130002]] : Dense(10,70,100,50,30,3) ,mse, epochs=500, batch_size=2
[[10.00632     1.8747585   0.07100728]] : Dense(10,70,100,50,30,3) ,mae, epochs=10000, batch_size=2
[[10.615263    2.0696187  -0.19277394]] : Dense(10,70,100,50,30,3) ,mae, epochs=500, batch_size=2
[[9.998037e+00 1.899733e+00 2.086075e-03]] : Dense(5,10,37,45,100,3) ,mse, epochs=1000, batch_size=1
'''


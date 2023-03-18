# x는 3개 y는 2개 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]) #(3,10)
x = x.T #(10,3)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]) # (2,10)
y = y.T # (10.2) 

# Q.[실습] 예측 [[9, 30, 210]] -> 예상 y값 [[10, 1.9]]

#2. 모델구성
model = Sequential()
model.add(Dense(15, input_dim=3)) #x값(10,3) input_dim 열의 개수에 맞춰야함 -> '3'
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(2)) # y값(10,2) output_dim 열의 개수에 맞춰야함 -> '2'
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=10, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print("loss :", loss)

result = model. predict([[9, 30, 210]])
print("[9, 30, 210]의 예측값 : ", result)

# [9, 30, 210]의 예측값 :  [[10.057739   1.9102501]] : Dense(30,10,50,33,30,2), mse, epochs=1000, batch_size=3
# [9, 30, 210]의 예측값 :  [[9.38773   1.1331272]] : Dense(10,30,50,30,20,2), mse, epochs=1000, batch_size=3
# [9, 30, 210]의 예측값 :  [[9.999999  1.8999937]] : Dense(10,30,100,50,20,2), mse, epochs=1000, batch_size=2
# [9, 30, 210]의 예측값 :  [[10.404082   1.7924371]] : Dense(10,30,100,50,20,2), mae, epochs=1000, batch_size=2
# [9, 30, 210]의 예측값 :  [[9.182166  1.8022105]] : Dense(15,30,50,30,10,2), mae, epochs=1000, batch_size=2
# [9, 30, 210]의 예측값 :  [[10.020496   1.8206719]] : Dense(15,30,50,30,10,2), mse, epochs=1000, batch_size=2
# [9, 30, 210]의 예측값 :  [[9.992395  1.8960314]] : Dense(15,30,50,40,20,2), mse, epochs=1000, batch_size=2
# [9, 30, 210]의 예측값 :  [[10.000014   1.9000062]] : Dense(15,30,100,40,20,2), mse, epochs=2000, batch_size=1
# [9, 30, 210]의 예측값 :  [[10.166225   1.9653583]] : Dense(15,70,100,60,20,2), mse, epochs=1000, batch_size=1

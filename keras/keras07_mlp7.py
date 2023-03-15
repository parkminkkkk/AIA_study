# x는 1개 y는 3개 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)]) #(1,10)
x = x.T #(10,1)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]]) # (3,10)
y = y.T # (10.3) 

# Q.[실습] 예측 [[9]] -> 예상 y값 [[10, 1.9, 0]]

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1)) #x값(10,1) input_dim 열의 개수에 맞춰야함 -> '1'
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(3)) # y값(10,3) output_dim 열의 개수에 맞춰야함 -> '3'
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y) 
#훈련데이터(x,y)로 평가를 한다는 게 맞는 것일까? 
#-> 훈련된 데이터는 값이 잘 나올 수 밖에 없으므로, 평가에 사용하면 안됨
#-> 객관적인 평가를 위해 훈련데이터는 평가에 사용하면 안됨 
#=> 따라서, 데이터를 나눈다 [훈련사용데이터 / 훈련사용x 데이터를 평가에 사용 : ex)1o개 데이터 중 7개 훈련, 남은 3개 평가]
#-> 통상적으로, 훈련한 데이터보다 결과데이터가 더 좋게 나오지 않는다.
print("loss :", loss)

result = model. predict([[9]])
print("[9]의 예측값 : ", result)


'''
[9]의 예측값 -> y값 [[10, 1.9, 0]]
[[1.0000000e+01 1.9000001e+00 5.6624413e-07]] : Dense(50,40,30,40,50,3), mse, epochs=100, batch_size=1
[[1.0001654e+01  1.8995094e+00 -1.1046976e-03]] : Dense(50,100,300,400,500,3), mse, epochs=100, batch_size=1
[[1.0000156e+01 1.9002384e+00 2.0049624e-03]] : Dense(10,20,40,100,80,3), mse, epochs=500, batch_size=2
[[10.114825    1.9273618  -0.44798318]] : Dennse(10,20,40,100,80,3), mae, epochs=500, batch_size=2
[[10.023349   1.8811125 -0.2116823]] : Dennse(10,20,40,60,80,3), mae, epochs=500, batch_size=2
[[9.9999943e+00 1.9000014e+00 4.7683716e-07]] : Dennse(10,20,40,60,80,3), mse, epochs=1000, batch_size=1
'''

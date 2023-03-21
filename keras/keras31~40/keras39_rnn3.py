import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM

#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터 : y없음 -> 만들어야 함 

x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])  
y= np.array([6,7,8,9,10])

print(x.shape, y.shape) #(5, 5) (5,)
#x의 shape = (행, 열, 훈련의 수(몇개씩 훈련)) => reshape해줘야함
#rnn통상 3차원데이터로 훈련시킴 / rnn구조 = 3차원 데이터 
x= x.reshape(5,5,1) #[[[1],[2],[3],[4]],[[2],[3].[4],[5] .....]]
print(x.shape) #(5, 5, 1)


#2. 모델구성 
model = Sequential()
model.add(SimpleRNN(32, input_shape=(5,1), activation='linear'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear')) 

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
import time
start = time.time()
model.fit(x, y, epochs=1000)
end = time.time()


#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #[[[7],[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(4,1))
print(x_predict.shape) #(1, 5, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[6,7,8,9,10]의 결과:', result)
print("time:", round(end-start, 2))

'''
#[[11]]만들기 
loss: 8.702943887328729e-05
[7,8,9,10]의 결과: [[10.541617]]

*데이터5
loss: 0.0
[6,7,8,9,10]의 결과: [[10.406439]]

-rnn기본 디폴트 함수 'tanh' -> 'linear'로 바꿔줌
loss: 2.028173115564691e-11
[6,7,8,9,10]의 결과: [[11.0030155]]

loss: 6.411937820971492e-12
[6,7,8,9,10]의 결과: [[11.040499]]
loss: 7.730705181910325e-13
[6,7,8,9,10]의 결과: [[11.009656]]
'''

'''
*cpu-rnn
Epoch 1000/1000
loss: 1.7280399116845202e-12
[6,7,8,9,10]의 결과: [[11.000001]]
time: 3.75

*gpu-rnn
Epoch 1000/1000
loss: 3.6379788613018216e-13      
[6,7,8,9,10]의 결과: [[10.999999]]
time: 41.78

#데이터가 커질 경우 gpu가 더 빠른 경우가 생길 수 도 있음
'''
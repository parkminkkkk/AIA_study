import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM

#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터 : y없음 -> 만들어야 함 

x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]])  
y= np.array([5,6,7,8,9,10])

print(x.shape, y.shape) #(6, 4) (6,)
#x의 shape = (행, 열, 훈련의 수(몇개씩 훈련)) => reshape해줘야함
#rnn통상 3차원데이터로 훈련시킴 / rnn구조 = 3차원 데이터 
x= x.reshape(6,4,1) #[[[1],[2],[3],[4]],[[2],[3].[4],[5] .....]]
print(x.shape) #(6, 4, 1)


#2. 모델구성 
model = Sequential()
model.add(SimpleRNN(8, input_shape=(4,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32))
model.add(Dense(4, activation='relu'))
model.add(Dense(2))
model.add(Dense(1)) 

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000)

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([7,8,9,10]).reshape(1,4,1) #[[[7],[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(4,1))
print(x_predict.shape) #(1, 4, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[7,8,9,10]의 결과:', result)

'''
#[[11]]만들기 
loss: 8.702943887328729e-05
[7,8,9,10]의 결과: [[10.541617]]

'''


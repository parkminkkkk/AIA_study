# Rnn의 단점 -> LSTM사용
# 깊어진만큼 Gradient Vanishing(기울기소실) 문제/  Exploding(폭발) 문제가 발생 가능
# 길어진 데이터를 처리하면서 Input data의 초기 타임 스텝을 점점 잊어버림 

#히든부분만 다르고, input/output부분 똑같음 
#'timesteps를 어떻게 자를 것 인가' 가장 중요

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout

#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터 : y없음 -> 만들어야 함 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])  
#x : 훈련 시킬 데이터,input 
#[8,9,10]까지 x데이터에 넣으면 y데이터 없음, 바로 예측데이터로 가야하므로 xx
y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)
###x의 shape = (행, 열, 훈련의 수(몇개씩 훈련하는지)) => reshape해줘야함###
#rnn은 통상 3차원데이터로 훈련시킴 / rnn구조 = 3차원 데이터 
x= x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3].[4], .....]]
print(x.shape) #(7, 3, 1)


#2. 모델구성 
model = Sequential()
model.add(LSTM(16, input_shape=(3,1))) #[batch, / timesteps, feature]   
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

#rnn : 3차원 데이터 입력해서 2차원으로 출력함 
#따라서, rnn은 layer를 쌓지 못함 3차원 받아서 2차원 출력하므로.. => 다른 방법


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([8,9,10]).reshape(1,3,1) #[[[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(3,1))
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[8,9,10]의 결과:', result)

'''
#[[11]]만들기 
*simpleRNN
loss: 0.015219539403915405
[8,9,10]의 결과: [[10.759624]]

*LSTM
loss: 0.00018726162670645863
[8,9,10]의 결과: [[10.818505]]
#성능은 통상적으로 LSTM이 더 좋음 
'''


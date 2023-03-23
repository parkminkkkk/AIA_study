# return_sequences : LSTM에서 다음층으로 2차원이 아닌 3차원으로 던져준다 => 연속된 층에 LSTM사용 가능해짐 
# return_sequences사용 -> LSTN,GRU 연속해서 사용할때 씀


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout

#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) #(13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape) #(13,3,1)

#2. 모델구성 
model = Sequential()
model.add(LSTM(16, input_shape=(3,1), return_sequences=True, activation='linear')) #[batch, / timesteps, feature]   
model.add(LSTM(16, return_sequences=True))
model.add(SimpleRNN(8,return_sequences=True))
model.add(GRU(32))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([50,60,70]).reshape(1,3,1) 
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[50,60,70]의 결과:', result)


'''
*LSTM 
loss: 0.0007800068706274033
[50,60,70]의 결과: [[77.51989]]

*LSTM_linear
loss: 0.00022526454995386302
[50,60,70]의 결과: [[80.38856]]

*save_load
loss: 0.00044784494093619287
[50,60,70]의 결과: [[80.7304]]

*LSTN2/ return_sequence
loss: 0.0016091702273115516
[50,60,70]의 결과: [[76.26181]]

*LSTN2, rnn, GRU/ return_sequence*
loss: 0.003227110719308257
[50,60,70]의 결과: [[72.16763]]
'''

#(None,3,11) : hidden에서 던져준 값이 시계열 데이터가 아닌 랜던값을 던져 줌
#시계열 상태로 던져준다면 성능이 좋아지겠지만, 랜덤값을 던져주므로 성능이 더 안 좋아 질 수도 있다. 
#따라서, 통상적으로 2개 이상 연속으로 사용하면 성능이 더 안좋아짐
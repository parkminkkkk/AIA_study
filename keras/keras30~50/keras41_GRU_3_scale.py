#scale : 범위가 다를때 데이터를 맞춰봐라 

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, GRU

#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


# [실습] 80 만들기  <= # x_predict = np.array([50,60,70])
print(x.shape, y.shape) #(13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape) #(13,3,1)

#2. 모델구성 
model = Sequential()
model.add(GRU(16, input_shape=(3,1), activation='linear')) #[batch, / timesteps, feature]   
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1)) 

#모델 로드
# model = load_model('./_save/MCP/keras40_save_model.h5') 
# model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#모델 저장
# model.save('./_save/MCP/keras40_save_model.h5') 


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

*GRU_linear
loss: 0.0019464625511318445
[50,60,70]의 결과: [[79.97644]]
'''
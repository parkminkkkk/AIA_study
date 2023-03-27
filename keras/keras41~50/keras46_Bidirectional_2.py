import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, GRU, Bidirectional

#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


# [실습] 80 만들기  - Bidirectional추가
print(x.shape, y.shape) #(13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape) #(13,3,1)

#2. 모델구성 

model = Sequential()
model.add(Bidirectional(GRU(16, activation='linear'), input_shape=(3,1))) 
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#모델 로드
# model = load_model('./_save/MCP/keras40_save_model.h5') 
# model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)

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
loss: 0.0006137349409982562
[50,60,70]의 결과: [[81.54205]]
'''
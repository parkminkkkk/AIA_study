# return_sequences : LSTM에서 다음층으로 2차원이 아닌 3차원으로 던져준다 => 연속된 층에 LSTM사용 가능해짐 
# return_sequences사용 -> LSTM,GRU 연속해서 사용할때 씀


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
model.add(LSTM(10, input_shape=(3,1), return_sequences=True)) #[batch, / timesteps, feature]   
model.add(LSTM(11, return_sequences=True))
model.add(SimpleRNN(11, return_sequences=True))
model.add(SimpleRNN(11, return_sequences=True))
model.add(GRU(11))
model.add(Dense(1)) 

model.summary()

#rnn : 3차원 데이터 입력해서 2차원으로 출력함 
#따라서, rnn은 layer를 쌓지 못함 3차원 받아서 2차원 출력하므로.. => 다른 방법(return_sequences)
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 3, 10)             480       #(None, 3, 1) -> (None, 3, 10)

 lstm_1 (LSTM)               (None, 3, 11)             968       #(None, 3,10) -> (None, 3, 11)

 simple_rnn (SimpleRNN)      (None, 3, 11)             253

 simple_rnn_1 (SimpleRNN)    (None, 3, 11)             253

 gru (GRU)                   (None, 11)                792      #(None, 3,11) -> (None, 11)

 dense (Dense)               (None, 1)                 12

=================================================================
Total params: 2,758
Trainable params: 2,758
Non-trainable params: 0
_________________________________________________________________

'''

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([50,60,70]).reshape(1,3,1) #[[[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(3,1))
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[50,60,70]의 결과:', result)

'''
#[[11]]만들기 
*simpleRNN
loss: 0.015219539403915405
[8,9,10]의 결과: [[10.759624]]

*LSTM
loss: 0.00018726162670645863
[8,9,10]의 결과: [[10.818505]]
#성능은 통상적으로 LSTM이 더 좋음 

*GRU
loss: 0.00046220290823839605
[8,9,10]의 결과: [[10.520404]]
'''


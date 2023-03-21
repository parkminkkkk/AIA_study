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
model.add(SimpleRNN(10, input_shape=(5,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1)) 

model.summary()

'''
Total params = recurrent_weights + input_weights + biases

= (num_units*num_units)+(num_features*num_units) + (1*num_units)

= (num_features + num_units)* num_units + num_units

=>즉, ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)
----------------------------------------------------------------------------------------
RNN 레이어의 파라미터 수: (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim
----------------------------------------------------------------------------------------
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120          #10*10 + 10*1 +10 = 120
_________________________________________________________________  ___________________________
dense (Dense)                (None, 7)                 77           #7*10 + 7 = 77
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 8            #1*7 + 1 = 8
=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0
_________________________________________________________________
'''


# #3. 컴파일, 훈련 
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1000)

# #4. 평가, 예측 
# loss = model.evaluate(x, y)
# x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #[[[7],[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(4,1))
# print(x_predict.shape) #(1, 5, 1)

# result = model.predict(x_predict)
# print('loss:', loss)
# print('[6,7,8,9,10]의 결과:', result)

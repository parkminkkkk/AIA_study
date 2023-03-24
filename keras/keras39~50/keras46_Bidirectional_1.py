#Bidirectional(양방향) : 양방향 학습으로 성능 높임

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.layers import Bidirectional

#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터 : y없음 -> 만들어야 함 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])  

y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)
x= x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3].[4], .....]]
print(x.shape) #(7, 3, 1)


#2. 모델구성 
model = Sequential()
# model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1)))  #(랩핑모양 주의) 첫번째 파라미터에 모델/ 두번째 파라미터에 input_shape
model.add(Bidirectional(SimpleRNN(10, return_sequences=True), input_shape=(3,1))) 
model.add(LSTM(10, return_sequences=True))
model.add(Bidirectional(GRU(10)))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 bidirectional (Bidirectiona  (None, 20)               240          # 120*2(SimpleRnn) / 960 = 120*4*2 (LSTM)
 l)

 dense (Dense)               (None, 1)                 21

=================================================================
Total params: 261
Trainable params: 261
Non-trainable params: 0
_________________________________________________________________
'''
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten  

model = Sequential()                   #(N, 3)
model.add(Dense(10, input_shape=(3,))) #(batch_size(units), input_dim) #(N, 10)
model.add(Dense(units=15))             #출력 (batch_size, units) / units: output 노드의 개수 


model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 15)                165
=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0
_________________________________________________________________
'''

#(N, 3) : (batch_size, input_dim).
#units : 정수값으로, 현재 밀집층에 존재하는 뉴런의 개수를 지정 / 아웃풋 노드의 개수 
#units :  output 노드의 개수 



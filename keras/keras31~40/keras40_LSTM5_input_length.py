import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM


#2. 모델구성 
model = Sequential()                        
# model.add(LSTM(10, input_shape=(5,1)))          #[batch, / timesteps, feature]
model.add(LSTM(10, input_length=5, input_dim=1))  #[batch, input_length, input_dim]
#embedding할 때 input_length/dim분리해야 하는 경우 생길 수 있다/ input_lenth만 명시하는 경우도 있음(feature영향없이 timesteps만 사용할 때) 
# model.add(LSTM(10), input_dim=1, input_length=5) #가능은 하나 가독성 떨어짐 
 
model.add(Dense(7, activation='relu'))
model.add(Dense(1)) 

model.summary()

'''
#파라미터값 동일함 

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10)                480       

 dense (Dense)               (None, 7)                 77        

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0
_________________________________________________________________
'''
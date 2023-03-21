#히든부분만 다르고, input/output부분 똑같음 
#'timesteps를 어떻게 자를 것 인가' 가장 중요

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM


#2. 모델구성 
model = Sequential()                           
model.add(LSTM(10, input_shape=(5,1)))  #[batch, / timesteps, feature]    
model.add(Dense(7, activation='relu'))
model.add(Dense(1)) 

model.summary()



'''
params = 4 * ((size_of_input + 1) * size_of_output + size_of_output^2)
Param = 4*((input_shape_size +1) * ouput_node + output_node^2)
      = 4*[units(units+features) + 1(bias/units)]  
      = 4*[(10*10) + (10*1) + (10)]
-----------------------------------------------------------------
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480            #120(simpleRNN) * 4 

 dense (Dense)               (None, 7)                 77

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0
_________________________________________________________________
'''
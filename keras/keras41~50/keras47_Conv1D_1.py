import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

#1. 데이터 
#2. 모델 

model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))  #Total params: 541   /#unit

model.add(Conv1D(10,2,input_shape=(3,1))) #Total params: 141   =>conv가 속도 훨씬 빠르다 
#filter, kernel_size                      #(파라미터값 Conv2D계산과 동일)
model.add(Conv1D(10,2))                   #Total params: 301 (conv1D 2개 씀)
model.add(Flatten())        
model.add(Dense(5)) #unit
model.add(Dense(1))

model.summary()


#3차원 데이터 받아들임 => 특성을 추출하는 역할
#(LSTM, Conv1D-3차원)
#LSTM보다 Total Params양이 더 적기 때문에 Conv1D가 속도 더 빠름 
#Conv2D와 Conv1D 성능 유사, 더 좋을 수도 있음 



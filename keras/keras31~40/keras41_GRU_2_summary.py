#히든부분만 다르고, input/output부분 똑같음 
#'timesteps를 어떻게 자를 것 인가' 가장 중요

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU


#2. 모델구성 
model = Sequential()                           
model.add(GRU(10, input_shape=(5,1)))  #[batch, / timesteps, feature]    
model.add(Dense(7, activation='relu'))
model.add(Dense(1)) 

model.summary()



'''
#rnn  120 / #LSTM 480/ #GRU  390 

#초창기GRU모델 : 한개 게이트 줄임 =>360param /현재 30개 늘어남 -> 390
#gate 수 2개(reset gate(output대체), update gate(forget+input))

#  params = 3 [units * (feature+bias(2)+units)] 
# 텐서플로 최신버전에서 bias를 2로 인식  
= 3*[(10*10)+(10*2)+(10*1)]
 


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 10)                390       

 dense (Dense)               (None, 7)                 77        

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 475
Trainable params: 475
Non-trainable params: 0
_________________________________________________________________

'''

'''
GRU층의 모델 파라미터 개수를 계산해보면 GRU 셀에는 3개의 작은 셀이 존재한다. 
각각의 작은 셀에는 입력과 은닉 상태에 곱하는 가중치와 절편이 존재한다. 
입력에 곱하는 가중치는 1x10=10개 이고, 은닉 상태에 곱하는 가중치는 10x10=100개 이다. 
그리고 절편은 뉴런마다 하나씩이므로 10개이다. 모두 더하면 120개로 이런 작은 셀이 3개 있으니 모두 360개의 모델 파라미터가 필요하다. 
하지만 위의 결과를 보면 24개가 더 있는 것을 확인할 수 있다.

텐서플로에 기본적으로 구현된 GRU셀의 계산은 앞선 계산법과 좀 다르다. 
이전에는 셀의 출력과 은닉 상태가 곱해지는 것이 순서였다.
하지만 텐서플로에서는 은닉 상태가 먼저 가중치와 곱해진 다음 셀의 출력과 곱해진다. 
그래서 별로도 나눠서 표기하기 때문에 작은 셀마다 하나씩 절편이 추가되고 8개의 뉴런이 있으므로 24개의 모델 파라미터가 더해진다.

3[( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)]


'''
#1. 데이터
import numpy as np
x = np.array([1, 2, 3]) #[1,2,3]을 하나의 덩어리로 봄, 노드하나에 [1,2,3] 한 덩어리 들어감
y = np.array([1, 2, 3]) #input_dim=1

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential #Sequential :순차적모델
from tensorflow.keras.layers import Dense #Dense : 단순하게 연결된 모델 y=wx+b

model = Sequential() # 모델이름을 Sequential이라고 할거야
model.add(Dense(3, input_dim=1)) #input layer
 #(Dense(output개수,input_dim개수))/input_dim개수가 노드의 개수다..
model.add(Dense(4)) #model.add : 모델을 층층이 쌓아올림 (4:아웃풋-층별로, 상위가 인풋일 경우 명시하지 않아도 됨x)
model.add(Dense(5)) #hidden layer
model.add(Dense(3)) #hidden layer
model.add(Dense(1)) #output layer

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #compile : 기계어로 바꾸다
# mse : 평균제곱오차 , loss값은 음수 없음(거리의 차이므로 음수존재x)
# loss값은 항상 상대적
model.fit(x,y, epochs=100) #fit : 훈련시키다 (x,y의 데이터를 놓고, epochs:몇번 훈련 시킬것인가)



# loss: 0.0160
# loss: 0.0013
# loss: 2.0332
# loss: 0.0284


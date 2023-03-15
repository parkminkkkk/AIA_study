#1. 데이터 2. 모델구성 3. 컴파일,훈련 4. 평가,예측 - 이 순서대로 코딩하면 웬만한것은 가능 

#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y) #evaluate : 3번까지로 가중치 생성된것들을 평가하다
# 현재는 x,y값 넣어서 훈련한 값을 넣지만, 나중에는 훈련하지 않은 값 넣어서 평가함. 더 정확해짐(난도올라감)
print("loss : ", loss)

result = model. predict([4]) #predict : 예측하다
print("[4]의 예측값 : ", result)
# 값이 4에 가까워질 수 있도록 변화주면서 실행시켜보기 
# 'mse' 'mae', epochs값, node값, layer개수 추가/삭제 등등 변화줄 수 있음 


#[4]의 예측값 :  [[3.9854996]] : mae, epochs=200
#[4]의 예측값 :  [[3.9999974]] : mse, epochs=1000 add(Dense(3,9,7,5,4,3,1))
#loss :  0.002088489942252636/ [4]의 예측값 :  [[3.923916]] : mse, epochs=50 
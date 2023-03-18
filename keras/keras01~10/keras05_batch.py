import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#batch : 일괄작업,  batch단위로 쪼개서 작업
#[낮을수록 좋다(batch=1) 그러나,예외 존재] 
#-> 훈련양이 많아짐 -> 성능이 좋아질 수 있다 
#그러나, 속도가 느려짐, 오래걸림(단위만큼 x배 더 하는 거니까)
#batch size : 디폴트값 32

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([6])
print("[6]의 예측값 : ", result)

#API를 땡겨온다 -> 가독성을 좋게 하기 위해서 (이런것들을 위주로 하겠구나)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([6])
print("[6]의 예측값 : ", result)

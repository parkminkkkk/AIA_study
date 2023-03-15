import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[[1,2], [3,4], [5,6]],
              [[1,2], [4,5], [7,8]]])
y = np.array([[[5,6]]])
print(x.shape) #(2,3,2)
print(y.shape) #(1,2)
y = y.T
print(y.shape) #(2,1)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model. evaluate(x,y)
print("loss :", loss)

result = model. predict([[7,8]])
print("[7,8]의 예측값 : ", result)

# 왜 안되는거죠..?

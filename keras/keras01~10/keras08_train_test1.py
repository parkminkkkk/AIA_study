#훈련데이터(x,y)로 평가를 한다는 게 맞는 것일까? 
#-> 훈련된 데이터는 값이 잘 나올 수 밖에 없으므로, 평가에 사용하면 안됨
#-> 객관적인 평가를 위해 훈련데이터는 평가에 사용하면 안됨 
#=> 따라서, 데이터를 나눈다 [훈련에 사용하는 데이터 / 훈련사용하지 않은 데이터를 평가에 사용]
#-> 통상적으로, 훈련한 데이터보다 결과 데이터가 더 좋게 나오지 않는다.
#ex) 1o개 데이터 중 7개 훈련, 남은 3개 평가

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1]) 
# y = np.array([10,9,8,7,6,5,4,3,2,1,]) 제일 끝에 부분에는 ',' 있어도 괜찮음(error안 뜸)
# print(x)
# print(y)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

result = model. predict([11])
print("[11]의 예측값 : ", result)

'''
loss / [11]의 예측값 -> 11
0.023074263706803322 / [[10.771966]] : Dense(10,9,7,5,3,1), mse, epochs=100, batch_size=2
0.04138247296214104  / [[11.0461855]] : Dense(10,9,7,5,3,1), mae, epochs=100, batch_size=2
'''

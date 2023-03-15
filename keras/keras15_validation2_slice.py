from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터 
x_train = np.array(range(1,17)) #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
y_train = np.array(range(1,17))

print(x_train)
print(y_train)

# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])
#  # train + val = 전체 train데이터라 할 수 있음. (13개의 데이터중 10개는 train, 3개는 validation)
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])


#[실습] 슬라이싱! 
#x_val = x_train[14:17] # [15,16]
#print(x_val)
x_val = x_train[13:] # [14, 15,16]
y_val = y_train[13:] # [14, 15,16]
x_test = x_train[10:13] # [11,12,13]
y_test = y_train[10:13] # [11,12,13]



'''
#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='linear', input_dim=1))  # x의 shape (10, )
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1)) #y열 1


#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_data=(x_val, y_val))              #-> val_loss값 나옴


#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([17])
print('17의 예측값 : ', result)
'''
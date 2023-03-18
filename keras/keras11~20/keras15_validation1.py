# validation추가 
# train data set 70% 완벽한 데이터 구조 xx - > 일부 데이터 검증(validation)
# 훈련데이터 중 일부를 검증한다. (훈련하는 중에=model.fit)
# 즉, 데이터를 train, validation(train data일부) , test 세개로 나눈다 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터 
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
 # train + val = 전체 train데이터라 할 수 있음. (13개의 데이터중 10개는 train, 3개는 validation)
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])


#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='linear', input_dim=1))  # x의 shape (10.1)
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
1. loss : 0.005812821444123983/ 17의 예측값 :  [[16.84329]]
-Dense(5,4,3,2,1), epochs=100, batch_size=1, mse
2. loss : 0.028708139434456825/ 17의 예측값 :  [[17.04042]]
-Dense(5,4,3,2,1), epochs=100, batch_size=1, mae
3. loss : 0.0183283481746912/ 17의 예측값 :  [[17.029312]]
-Dense(5,4,3,2,1), epochs=1000, batch_size=1, mae
4. loss : 0.002132415771484375/ 17의 예측값 :  [[16.994324]]
-Dense(5,4,3,2,1), epochs=3000, batch_size=1, mae
5. loss : 0.03007030487060547/ 17의 예측값 :  [[17.040249]]
-Dense(16,8,4,2,1), epochs=1000, batch_size=1, mae

'''
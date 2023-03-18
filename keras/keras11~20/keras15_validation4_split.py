#데이터 train,test로 한 이후, 훈련과정에서 validation비율 정해줌 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터 
x = np.array(range(1,17))
y = np.array(range(1,17))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2, random_state=123)


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
          validation_split=0.2)              #검증 0.2(20%)로 할 것 


#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([17])
print('17의 예측값 : ', result)
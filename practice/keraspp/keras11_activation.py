#Q. [실습] 
# 1. train_size 0.7
# 2. R2 0.8이상
# activation : 활성화함수 'relu' 'sigmoid' : 직선함수에 곡선을 줘서 랜덤한 데이터 값의 정확도를 올려줌 

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=1)

#2. 모델구성 
model = Sequential()
model.add(Dense(16, input_dim=13, activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=3000, batch_size=16)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

'''
[R2 0.8이상, train_size=0.7]
1. loss : 10.342726707458496/ r2스코어 :  0.8871552610326644
activation='relu', train_size=0.7, random_state=1, Dense(16,9,7,5,3,1), epochs=3000, batch_size=16

'''
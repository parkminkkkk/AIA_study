#Q. [실습] 
# 1. train_size 0.7
# 2. R2 0.8이상
# dropout : 0부터 1 사이의 확률로 뉴런을 제거(drop)하는 기법
# 랜덤하게 일부 뉴런을 제거하여 모델이 특정 뉴런에 과도하게 의존하지 않도록 하는 정규화 기법

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=4)

#2. 모델구성 
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dense(1, activation='linear'))

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
-dropout은..?
1. loss : 19.240880966186523 / r2스코어 :  0.7619530502282688
train_size=0.7, shuffle=True, random_state=123
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
2. loss : 47.681217193603516 / r2스코어 :  0.5433711499804879
train_size=0.7, shuffle=True, random_state=4
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dense(1, activation='linear'))
'''
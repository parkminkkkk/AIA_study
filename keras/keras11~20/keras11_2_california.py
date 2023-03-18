#Q.[실습]
#조건1 : train_size=0.7~0.9 사이로
#R2 0.55~0.6 이상 
#(데이터 맞기 어렵다는 뜻 : 실무에서는 더함 -> 데이터 정제를 잘해야함 )

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=4)


#2. 모델구성 
model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=30000, batch_size=1000)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)



'''
[R2 0.55~0.6 이상]
1. loss : 0.6252856254577637 / r2스코어 :  0.5308127558687077 
train_size=0.7, random_state=1234, Dense(13,30,50,10,100,40,1), mse, epoch=500, batch_size=32

2. loss : 0.6020143032073975 / r2스코어 :  0.5529008043107493
train_size=0.8, random_state=124, Dense(10,50,10,13,10,30,1), mse, epoch=1000, batch_size=206
3. loss : 0.5374529361724854/ r2스코어 :  0.5687972592338953
train_size=0.8, random_state=124, Dense(10,50,70,100,13,10,30,1), mae, epoch=2064, batch_size=206
4. loss : 0.5147770643234253/ r2스코어 :  0.5908028105514126
train_size=0.9, random_state=4, Dense(32,16,8,4,1), mse, epoch=30000, batch_size=1000
5.loss : 0.5261240601539612/ r2스코어 :  0.5973490330272526
train_size=0.7, random_state=4, Dense(32,16,8,4,1), mse, epoch=30000, batch_size=1000

'''


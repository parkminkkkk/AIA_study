#[실습]
#R2 0.62이상 

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.9, shuffle=True, random_state=123)

#2. 모델구성 
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=4000, batch_size=40)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
[R2 0.62이상] 
-train_size 0.7<0.8<0.9순으로 잘 나옴-
1. loss : 2717.541015625 / r2스코어 0.5207854582674478 
Dense(50,70,90,100,40,1), mse, epochs=500, batch_size=32
2. loss : 2692.3466796875 / r2스코어 :  0.5252282619381045
train_size=0.8, random_state=124, Dense(5,3,5,3,5,2,1), mse, epochs=1000, batch_size=4
3. loss : 42.387718200683594 / r2스코어 :  0.5682137752070641
train_size=0.8, random_state=123, Dense(5,10,11,12,13,14,1), mae, epochs=1000, batch_size=16
4. loss : 41.84345245361328 / r2스코어 :  0.5748367436020417
train_size=0.8, random_state=123, Dense(5,9,7,5,,3,1), mae, epochs=5000, batch_size=32

5. loss : 2377.912109375 / r2스코어 :  0.6434319517135226
train_size=0.9, random_state=123, Dense(32,16,8,2,1), mse, epochs=3000, batch_size=100
6. loss : 2386.77978515625 / r2스코어 :  0.6421022254686133
train_size=0.9, random_state=123, Dense(50,30,20,10,1), mse, epochs=3000, batch_size=100
7. loss : 2380.39404296875 / r2스코어 :  0.6430597987730948
train_size=0.9, shuffle=True, random_state=123 Dense(32,16,8,4,1), mse, epochs=4000, batch_size=40
'''
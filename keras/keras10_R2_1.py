#R2(결정계수) r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=124)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)


'''
# [검색]R2, R제곱, 결정계수
결정계수(Coefficient of Determination) (R²,R Square)
-모형의 적합도를 평가하는 다른 평가 지표 
-회귀모형 내에서 설명변수 x로 설명할 수 있는 반응변수 y의 변동 비율입니다.
-R²는 0부터 1사이의 값
-R²=0 : x와 y는 어떠한 선형 상관관계X
-R²=1 : x와 y는 완벽한 선형 상관관계ㅇ
-즉, 1에 가까울수록 선형상관관계가 크다 
-loss는 0에 가까울수록 좋다 -> loss와 R2 상호보완적으로 사용할 수 있다 
 만약, loss와 R2가 얽힌 경우(상반되게 나오는 경우)는 loss로 판단 (loss가 절대적, R2는 보조) 
'''
y_predict = model.predict(x_test) 
#x값에 훈련데이터(70%)까지 포함되어있음, 점수 더 좋게 나올 수 있음 -> 따라서, (x_test)사용! 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #원래 y값과 예측한 y값 비교함 
print('r2스코어 : ', r2)

'''
#r2스코어 :  0.7137633631009944 - 0.71 : 71%정도! 맞는다.. 확실한 71%(x) ->(x,y): 훈련+테스트 전체 범위
#r2스코어 :  0.7255624279704207 : (x_test,y_test)범위, Dense(50,40,35,20,10,1), mse, epochs=500, batch_size=1
[r2스코어 0.99까지 올려보기] - train사이즈, random_state 변경 가능!
loss : 6.787201404571533 / r2스코어 :  0.7529431158973166 : Dense(5,10,35,20,10,1), mse, epochs=1000, batch_size=1
loss : 6.292457580566406 / r2스코어 :  0.7709518596421572 : Dense(50,1000,350,200,100,1), mse, epochs=1000, batch_size=1
loss : 2.0124144554138184 / r2스코어 :  0.6301858274504479 : Dense(50,100,500,200,100,1), mae, epochs=1000, batch_size=1
loss : 0.8664274215698242 / r2스코어 :  0.9738723966313346 : train_size=0.7, random_state=124, Dense(50,100,500,200,100,1), mae, epochs=1000, batch_size=1
loss : 0.5655500888824463 / r2스코어 :  0.9888401990812203 : train_size=0.8, random_state=124, Dense(50,100,500,200,100,1), mae, epochs=1000, batch_size=1
loss : 0.9153837561607361 / r2스코어 :  0.9715370952783801 : train_size=0.8, random_state=124, Dense(5,70,50,30,10,1), mae, epochs=100, batch_size=1
loss : 1.3632367849349976 / r2스코어 :  0.9686612107566254 : train_size=0.8, random_state=124, Dense(5,70,50,30,10,1), mse, epochs=100, batch_size=1
loss : 0.6165845990180969 / r2스코어 :  0.9876734167663778 : train_size=0.8, random_state=124, Dense(5,90,70,50,30,1), mae, epochs=100, batch_size=1

'''

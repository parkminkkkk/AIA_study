#verbose : 말수가 많은 
#verbose = 0 : 'off', 아무것도 안나온다(아무정보 보여주지x)
#verbose = 1,'auto' : 'on',  디폴트값=1 (='auto'동일) (많은 정보량 보여줌, step별로 진행상황 보여줌)
#verbose = 2 : 프로그래스 바 보여주지 않음
#verbose = 3,4,5... : epoch만 보여줌 (3이상 동일= 0,1,2제외 동일)

#파라미터값 모를때 : keras.io / tensoflow.org 사이트내 검색가능
#kaggle.com : 개발자 커뮤니티(google) / 데이터 없을때 가져다쓰기 좋음 / 취업시 kaggle에서 얼마나 문제를 풀었느냐..
#Dacon.io : 한국판kaggle

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
        train_size=0.7, shuffle=True, random_state=650874)

#2. 모델구성 
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(16))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=50, verbose='auto')

#4.
y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

#r2값만 나옴 
#5/5 [==============================] - 0s 756us/step
#r2스코어 :  0.6336104365663701

'''
#4. 평가, 예측
loss = model.evaluate(x_test,y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)
'''
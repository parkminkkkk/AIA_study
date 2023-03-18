# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법! 
# Hint : 사이킷런 sklearn / scikit-learn
# train_test_split
"""
* train 하는 것의 문제점은?
평가하는 부분에서는 오차 적을 수 있으나, 예측값까지 갔을때에는 점점 더 오차가 커질 수 있음
즉, 범위밖에서는 오차가 커짐(=loss값이 나쁨), 명확하게 판단하기 어려움
=>따라서, 훈련을 시킬때에는 가급적 전체 범위내에서 훈련을 시키되, 그 훈련범위내에서 평가를 시킨다 (평가데이터에 포함은x)
  (랜덤하게 평가값을 빼기때문에 전보다 오차(loss) 줄일 수 있음)
=> train과 test를 분리하되, 전체 값에서 n%(일부의 비율)로 Test값 혹은 Train값을 뽑아 내야한다!
"""

'''
*구글링 정보*
1. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
train_test_split(x,y, test_size = 0.3, train_size = 0.7)
train_test_split(x,y, shuffle=False) 
# False : 랜덤 없이 순차적으로 데이터를 분리
# true : 랜덤 하겠다 

2. 
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
# stratify=y : 매개변수/ original dataset에서 특정 클래스 비율이 불균형한 경우,
  stratify 매개변수에 타깃 데이터를 지정하여 호출하면 (어떠한 통계적 기법을 통해서) 이 비율이 유지될 수 있도록 샘플링한다. 

'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, # train_size=0.7, : train,test_size 둘 중 하나만 적어도 적용가능
    test_size=0.3,
    random_state=1234, #랜덤의 난수표가 있음-그대로 나옴  
    shuffle=True,
    )
# 훈련시킬때마다 데이터 값이 바뀐다면(셔플로 인해) 잘 만든 모델(#2.모델구성)인지 확인 불가능함
# 따라서, 랜덤을 하더라도 데이터값을 고정해야함 => 잡아두는 것이 '씨드'임 random_state로 test값을 고정시킴!
# shuffle 디폴트 = True /  False로 잡으면 순서대로 나옴.. #shuffle=False   
print(x_train)  #[2 1 9 5 6 7 4]
print(x_test)   #[ 8  3 10]
print(y_train)  #[ 9 10  2  6  5  4  7]
print(y_test)   #[3 8 1]

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([11])
print("[11]의 예측값 : ", result)

'''
loss/ [11]의 예측값 
1.7405916452407837 / [[1.8098862]] : Dense(10,9,7,5,3,1), mse, epochs=100, batch_size=2
5.93175127505674e-06/[[0.00338175]] : Dense(10,90,70,50,30,1), mse, epochs=100, batch_size=2
'''





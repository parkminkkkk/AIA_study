#Q.[실습] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
# list표기법 [m(0) : n] : (0부터 시작) m번째~n-1번째 까지
# range(n)= 0 ~ n-1까지 : 시작숫자 0부터, 끝은 n-1 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

x_train = x[0:7] #0번째부터 6번쨰까지 = [0:7] = [ :7]  
x_test = x[7:10] #7번째부터 9번째까지 = [7:10] = [7: ]
y_train = y[:7]
y_test = y[7:]

#print(x_train) #[1,2,3,4,5,6,7]     *[0:7]
#print(x_test)  #[8,9,10]            *[7:0]
#print(y_train) #[1,2,3,4,5,6,7]     *[ :7]
#print(y_test)  #[8,9,10]            *[7: ]
#print(x_train.shape, x_test.shape)  #(7, ) (3, )
#print(y_train.shape, y_test.shape)  #(7, ) (3, )

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

result = model. predict([11])
print("[11]의 예측값 : ", result)


'''
loss / [11]의 예측값 -> y값 0
0.01752767711877823 / [[0.18850334]] : Dense(10,9,7,5,3,1), mse, epochs=100, batch_size=2
'''

"""
* train 하는 것의 문제점은?
평가하는 부분에서는 오차 적을 수 있으나, 예측값까지 갔을때에는 점점 더 오차가 커질 수 있음
즉, 범위밖에서는 오차가 커짐(=loss값이 나쁨), 명확하게 판단하기 어려움
=>따라서, 훈련을 시킬때에는 가급적 전체 범위내에서 훈련을 시키되, 그 훈련범위내에서 평가를 시킨다 (평가데이터에 포함은x)
  (랜덤하게 평가값을 빼기때문에 전보다 오차(loss) 줄일 수 있음)
=> train과 test를 분리하되, 전체 값에서 n%(일부의 비율)로 Test값 혹은 Train값을 뽑아 내야한다!
"""

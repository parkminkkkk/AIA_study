#RNN
# rnn구조 = 3차원 데이터 (input_shape=(n,n))
#시계열 데이터에 많이 쓰임 (y가 없다. 우리가 y 만들어야 함)
#과거의 데이터들이 현재에 영향을 미친다 (현재 연산이 다음 연산에 영향을 미침)
#즉, 과거의 값의 연산을 현재에 일부 더해주겠다. 
#상태값(state)이 다음 값에 load된다 (state값이 너무 크므로 activation해줌/activation default: 'tanh')
#state : -1 ~ 1사이 (작은 값을 던져주고 싶을때 음수값도 나올 수 있도록 하기위해 'sigmoid'아닌 'tanh'를 사용함) 
#weight값 두개 (y=wx+b / (hidden)state의 weight)

#단점) rnn이 너무 길어지면 문제 발생(정확도떨어짐) 
#즉, 짧은 시퀀스 처리에 유리/ 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀어지는 경우 학습 능력이 현저히 저하됨 
#왜냐하면, 20년전 데이터가 현재의 데이터에 영향을 미칠 확률은 적기때문에 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터 : y없음 -> 만들어야 함 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])  
#x : 훈련 시킬 데이터,input 
#[8,9,10]까지 x데이터에 넣으면 y데이터 없음, 바로 예측데이터로 가야하므로 xx
y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)
###x의 shape = (행, 열, 훈련의 수(몇개씩 훈련하는지)) => reshape해줘야함###
#rnn은 통상 3차원데이터로 훈련시킴 / rnn구조 = 3차원 데이터 
x= x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3].[4], .....]]
print(x.shape) #(7, 3, 1)


#2. 모델구성 
model = Sequential()
model.add(SimpleRNN(16, input_shape=(3,1))) #[batch, / timesteps, feature]   
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

#rnn : 3차원 데이터 입력해서 2차원으로 출력함 
#따라서, rnn은 layer를 쌓지 못함 3차원 받아서 2차원 출력하므로.. => 다른 방법


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([8,9,10]).reshape(1,3,1) #[[[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(3,1))
print(x_predict.shape) #(1, 3, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[8,9,10]의 결과:', result)

'''
#[[11]]만들기 
loss: 1.237274408340454
[8,9,10]의 결과: [[7.8994308]]

loss: 0.00026558313402347267
[8,9,10]의 결과: [[10.53814]]
loss: 0.015219539403915405
[8,9,10]의 결과: [[10.759624]]
'''


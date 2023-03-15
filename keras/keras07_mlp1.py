import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#10행 2열 (10,2) 10x2
#행 : 데이터의 개수 / 열 : 데이터의 특성, 컬럼, 피처
#**행무시, 열우선**
# 행개수 그닥 중요x, 열을 보고 모델링의 개수를 파악할 것임
# 행은 추가해도 크게달라지지 않지만, 열을 추가하면 완전 달라지므로..

#1. 데이터
x = np.array(
   [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],
    [7, 1.4],
    [8, 1.5],
    [9, 1.6],
    [10, 1.4]]  
)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)   #(10, 2) -> 2개의 특성을 가진 10개의 데이터 
print(y.shape)   #(10, )  -> 스칼라 10개, 벡터 1개 

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=2))  #열이 2개이므로, input_dim : 데이터의 특성, 컬럼, 피처의 개수, => 노드의 개수와 맞춰주면됨, 열의 수치와 동일 (노드가 2개로 시작)
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print("loss :", loss)
#모델의 평가는 model.fit에서 생성된 가중치로 한다. weight가 생성됨..
#최종적으로 가중치 완성되므로, evaluate마지막 값= loss값 


result = model. predict([[10, 1.4]])  
print("[10, 1.4]의 예측값 : ", result)
# 20이 나온다면 잘 만든 것 (y데이터값)
# *행무시, 열우선*-> 열의 수에 맞춰야하므로 1행 2열([[10, 1.4]])로 맞춰줘야 함!!  <-([10, 1.4]) (x)


# [10, 1.4]의 예측값 :  [[20.430218]]/ Dense(30, 15,7,5,1) epochs=30, batch_size=1  
# [10, 1.4]의 예측값 :  [[20.]] / Dense(30,15,7,5,1) epochs=10000, batch_size=1
# [10, 1.4]의 예측값 :  [[20.]] / Dense(30,15,7,5,1) epochs=1000, batch_size=2

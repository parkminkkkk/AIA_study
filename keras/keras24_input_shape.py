#input_dim의 문제점 
#이때까지 2차원 데이터만 함 (행,열)
# -> 만약 4차원 데이터라면? 이미지 데이터 (가로,세로,컬러,장수) 

#input_shape 사용 
#데이터가 3차원이면(통상 시계열 데이터 )
#(1000, 100, 1) -> input_shape(100, 1) #항상 제일 앞에 있는 것은 데이터의 개수임, 따라서 열은 (100,1)
#데이터가 4차원이라면(이미지 데이터)
#(60000, 32, 32, 3) -> input_shape(32,32,3) #컬러이미지가 6만장, 따라서, 행무시 열우선(32,32,3)


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets['target']
# print(type(x)) #<class 'numpy.ndarray'>
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=1234,)

#data scaling(스케일링)
scaler = MinMaxScaler() 
scaler.fit(x_train) #x_train범위만큼 잡아라
x_train = scaler.transform(x_train) #변환
#x_train의 변환 범위에 맞춰서 하라는 뜻이므로 scaler.fit할 필요x 
x_test = scaler.transform(x_test) #x_train의 범위만큼 잡아서 변환하라 

print(np.min(x_test), np.max(x_test)) 
#-0.0019120458891013214 1.1629424392720045
#범위 밖 데이터이므로, 성능이 더 좋아질 수 있다. 


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_shape=(13,))) #스칼라13개, 벡터1개 : 열의 개수를 벡터 형식으로 표시
# model.add(Dense(1, input_dim=13))

#input_shape
#데이터가 3차원이면(시계열 데이터)
#(1000, 100, 1) -> input_shape(100, 1) #항상 제일 앞에 있는 것은 데이터의 개수임, 따라서 열은 (100,1)
#데이터가 4차원이라면(이미지 데이터)
#(60000, 32, 32, 3) -> input_shape(32,32,3) #컬러이미지가 6만장, 따라서, 행무시 열우선(32,32,3)


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)



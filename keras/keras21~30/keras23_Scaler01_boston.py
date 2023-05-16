'''
boston import 안되는 경우(1.2부터 안됨)
pip uninstall scikit-learn
pip install scikit-learn==1.1
pip list => sklearn(x)
'''
# Scaler(스케일링기법)
# 1. 정규화(normalization) 
# 모든 데이터(x)를 0-1사이로 압축시킨다 / (x만 해당, y는 해당x)
# 변환시킨 x(0-1사이 압축)로 계산을 해도 y의 값은 동일하다.  (동일한 비율로 x가 변경되기때문에)
# 성능좋아질 수 있음, 속도 빨라짐 등등으 장점 존재하므로 정규화함 / 단점) 성능이 안좋은 경우도 존재한다..

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
#과제 : StandardScaler, MaxAbsScaler, RobustScaler
#전처리(preprocessing) **데이터 전처리 중요**

#1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets['target']
# print(type(x)) #<class 'numpy.ndarray'>
# print(x)

'''
print(np.min(x), np.max(x)) #0.0 711.0
scaler = MinMaxScaler()           #정의 
scaler.fit(x)                     #사용(변환) : 비율로 변환할 준비를 해 #MinMaxScaler() MinMaxScaler()
x = scaler.fit_transform(x)       #사용(변환) : 변환한걸로 바꿔줘 
print(np.min(x), np.max(x)) #0.0 1.0 
#통상적으로 전체를 정규화하지 않는다. (과적합발생할 수 있음)
#x_train만 정규화한다** (과적합을 막는 방법)
#따라서, 통상적으로 split진행한 후에 스케일링 진행한다! 
------------------------------------------------------------------
#data scaling(스케일링)
# scaler = StandardScaler()  : 표준정규 
# scaler = MaxAbsScaler()   
# scaler = RobustScaler()  
# scaler = MinMaxScaler() 
# 선택해서 사용할 수 있음 
'''

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=1234,)

#data scaling(스케일링)
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) #x_train범위만큼 잡아라
x_train = scaler.transform(x_train) #변환
#x_train의 변환 범위에 맞춰서 하라는 뜻이므로 scaler.fit할 필요x 
#x_train = scaler.fit_transform(x_train) #위에 두줄 한줄로 합침 fit_transform
x_test = scaler.transform(x_test) #x_train의 범위만큼 잡아서 변환하라 

print(np.min(x_test), np.max(x_test)) 
#-0.0019120458891013214 1.1629424392720045
#범위 밖 데이터이므로, 성능이 더 좋아질 수 있다. 


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)



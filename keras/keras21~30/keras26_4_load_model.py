#load_model 

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
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
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train) #위에 두줄 한줄로 합침 fit_transform
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test)) 


#2. 모델구성 (함수형모델) 

#모델 로드
model = load_model('./_save/keras26_3_save_model.h5')  #가중치 저장
model.summary()


#3. 컴파일, 훈련 
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=30) 

# model.save('./_save/keras26_1_save_model.h5')

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)




'''
*save : 모델만 저장할 뿐만아니라 가중치까지 저장하는 기능이 있다. 
*3_save파일 
-컴파일, 훈련 다음에 save했으므로 가중치까지 저장되었다! 
-weight가 고정되었으므로 값이 바뀌지 않는다! 

save1. 모델 구성 다음 save : 모델까지 저장됨
save2. 컴파일, 훈련 다음 save : 가중치까지 저장됨 
=> 즉, 위치에 따라 사용할 수 있다.  

#대회 : 소스, 가중치파일 요구함
'''




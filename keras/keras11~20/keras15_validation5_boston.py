#Q. [실습] boston파일 + validation 추가 


from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

'''
# print(x) #수치화된 것을 전처리한 데이터
# print(y)
# print(datasets) #feature name 

print(datasets. feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# 열의 특성, 피처, 컬럼이 13개 = input_dim = 13

print(datasets.DESCR)
# Number of Instances : 506 = 506개의 예시, 데이터의 개수 
# Number of Attributes :특성, 컬럼, 피처, 열 = 13 numeric/category 
# MEDV : 결과값(y값)

print(x.shape, y.shape)
# (506, 13) (506,) 
# input_dim=1, final output_dim =1
'''

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
model.fit(x_train,y_train, epochs=1000, batch_size=100,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
loss : 23.914884567260742
r2스코어 :  0.601263889621271
'''
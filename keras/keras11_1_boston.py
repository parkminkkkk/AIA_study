#Q. [실습] 
# 1. train_size 0.7
# 2. R2 0.8이상

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
model.fit(x_train,y_train, epochs=10000, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
[train_size=0.7일 때, R2 0.8이상]
-loss값, randon_state값 차이(고정), mse/mae의 차이-
1. loss : 4.080691814422607 / r2스코어 :  0.6317668075711564 
Dense(10,10,10,5,3,1), 'mse', epochs=100, batch_size=1
2. loss : 25.782730102539062 / r2스코어 :  0.7203000535470858
Dense(10,10,100,50,30,1) 'mse', epochs=300, batch_size=1
3. loss : 3.599179267883301 / r2스코어 :  0.6905777909599332
Dense(10,10,100,50,30,1) 'mse', epochs=300, batch_size=1
4. loss : 26.76950454711914 / r2스코어 :  0.7095951730363439 
Dense(13,10,50,90,40,1) 'mse', epochs=300, batch_size=1

5. loss : 17.071054458618164/ r2스코어 :  0.8099429373690612
random_state=2579, Dense(16,8,4,2,1) 'mse', epochs=10000, batch_size=100
6. loss : 17.444496154785156/ r2스코어 :  0.7091455394715729
random_state=650874, Dense(16,8,4,2,1) 'mse', epochs=10000, batch_size=100
7. loss : 17.553396224975586/ r2스코어 :  0.7073298512187399
random_state=650874, Dense(32,16,2,2,1) 'mse', epochs=10000, batch_size=100
'''
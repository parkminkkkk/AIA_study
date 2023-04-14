###회귀 데이터 모아서 테스트###
#회기
# 1. boston
# 2. california
# 3. ddarang
# 4. kaggle_bike

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#1. 데이터 
#1-1
datasets = fetch_california_housing()
x, y = fetch_california_housing(return_X_y=True)

#1-2
path = './_data/dacon_ddarung/'
path_save = './_save/dacon_ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #(1459, 10)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #(715, 9) count제외
###결측치제거### 
train_csv = train_csv.dropna() 
print(train_csv.isnull().sum())
# print(train_csv.info())
print(train_csv.shape)  #(1328, 10)

###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

#1-3
#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #(10886, 11)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #(6493, 8)  /casual(비회원)  registered(회원) count 3개 차이남
#결측치 없음
###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)
y = train_csv['count']
print(y)


xydata =[]
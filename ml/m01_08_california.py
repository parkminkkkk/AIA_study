###회귀 데이터 모아서 테스트###
#회기
# 1. boston
# 2. california
# 3. ddarang
# 4. kaggle_bike

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#1. 데이터 
#1-1
datasets = fetch_california_housing()
x, y = fetch_california_housing(return_X_y=True)
#1-2
path_ddarung = './_data/dacon_ddarung/'
path_kaggle_bike = './_data/kaggle_bike/'


ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
ddarung_test = pd.read_csv(path_ddarung + 'test.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle_bike + 'train.csv', index_col=0)
kaggle_test = pd.read_csv(path_kaggle_bike + 'test.csv', index_col=0)

data_list = [fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]
name_list = ['fetch_california_housing', 'load_diabetes', 'ddarung_train', 'kaggle_train']

model_list = [DecisionTreeRegressor(), RandomForestRegressor()]

for i in range(len(data_list)):
    if i<2:
        x, y = data_list[i](return_X_y=True) 
    elif i==2:
        x = ddarung_train.drop(['count'], axis=1)
        y = ddarung_train['count']
    elif i==3:
        x = kaggle_train.drop(['casual', 'registered', 'count'], axis=1)
        y = kaggle_train['count']
    for j in model_list:
        model = j
        model.fit(x, y)
        results = model.score(x, y)
        print(name_list[i], type(j).__name__, results)
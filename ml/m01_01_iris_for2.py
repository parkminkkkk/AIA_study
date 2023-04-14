###분류데이터들 모아서 테스트###
# #분류
# 1. iris 
# 2. cancer
# 3. dacon_diabets
# 4. wine
# 5. fetch_covtype
# 6. digits

import warnings
warnings.filterwarnings('ignore') 

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

#1. 데이터 
datasets = [load_iris,load_breast_cancer,load_wine,load_digits]
            #,fetch_covtype(return_X_y=True)]

#2. 모델구성
models = [RandomForestRegressor(),DecisionTreeRegressor(),LogisticRegression(),LinearSVC()]

scaler = [MinMaxScaler(),MaxAbsScaler(),RobustScaler(),StandardScaler()]

# Loop through the datasets and models, fit the models, and print the results
for i in datasets : 
    x, y = i(return_X_y=True)
    for j in scaler:
        x = j.fit_transform(x)
        for k in models : 
            model = k
            model.fit(x,y)
            results = model.score(x,y)
            print(i.__name__,type(j).__name__, type(k).__name__, results)






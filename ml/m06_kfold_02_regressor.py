# all_estimators : 모든 모델에 대한 평가 (회귀 55개 모델)

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators

import sklearn as sk
print(sk.__version__)    #1.0.2
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
datasets = [load_diabetes(return_X_y=True),
            fetch_california_housing(return_X_y=True)] 
dataname = ['diabetes','california']

# scaler = [MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#1. 데이터 
for i, v in enumerate(datasets):
    x, y = v

#2. 모델구성
    allAlgorithms = all_estimators(type_filter='regressor')
    max_score = 0
    max_name = 'max_model'
    
    for (name, algorithm) in allAlgorithms:
        try: #예외처리
            model = algorithm()

            scores = cross_val_score(model, x, y, cv=kfold)
            mean = round(np.mean(scores),4)
            # print('acc:', scores, '\ncross_val_score 평균:', mean)

            if max_score < mean: 
               max_score = mean
               max_name = name
        except:
            continue #continue: error 무시하고 계속 for문 돌리기 #break = for문 중단해라

    #dataset name , 최고모델, 성능
    print('========', dataname[i],'========')        
    print('최고모델:', max_name, max_score)
    print('================================')  


# ======== diabetes ========      
# 최고모델: ARDRegression 0.4801  
# ================================











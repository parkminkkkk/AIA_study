###분류데이터들 모아서 테스트###
# #분류
# 1. iris 
# 2. cancer
# 3. wine
# 4. fetch_covtype
# 5. digits

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
datasets = [load_iris(return_X_y=True),load_breast_cancer(return_X_y=True),load_wine(return_X_y=True),
            load_digits(return_X_y=True),fetch_covtype(return_X_y=True)]

#2. 모델구성
models = [RandomForestRegressor(),DecisionTreeRegressor(),
          LogisticRegression(),LinearSVC()]

scaler = MinMaxScaler()

# Loop through the datasets and models, fit the models, and print the results
for i, dataset in enumerate(datasets):
    x, y = dataset
    x = scaler.fit_transform(x)
    print(f"\nResults for dataset {i+1}:")
    for j, model in enumerate(models):
        model.fit(x, y)
        score = model.score(x, y)
        print(f"  Model {j+1}: {score:.3f}")


'''
#분류
Results for dataset 1: iris
  Model 1: 0.992
  Model 2: 1.000
  Model 3: 0.973
  Model 4: 0.967      

Results for dataset 2: cancer
  Model 1: 0.980
  Model 2: 1.000
  Model 3: 0.947
  Model 4: 0.926

Results for dataset 3: wine
  Model 1: 0.992
  Model 2: 1.000
  Model 3: 0.966
  Model 4: 0.876

Results for dataset 4: digits
  Model 1: 0.984
  Model 2: 1.000
  Model 3: 1.000
  Model 4: 0.992

Results for dataset 5: fetch_covtype
  Model 1: 0.991
  Model 2: 1.000
  Model 3: 0.619
  Model 4: 0.572
----------------
#회기
1. boston
2. california
3. ddarang
4. kaggle_bike
'''

'''
#아이리스 
#DL
0.17264685034751892
0.946666657924652

#ML
0.9666666666666667
'''







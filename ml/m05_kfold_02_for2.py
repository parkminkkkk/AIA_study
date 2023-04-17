#k-fold (교차검증) - 데이터 전처리과정
#분류
#iris, cancer, wine, digits, fetch_covtype

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') 

#1. 데이터 
datasets = [load_iris(return_X_y=True),load_breast_cancer(return_X_y=True),load_wine(return_X_y=True),
            load_digits(return_X_y=True),fetch_covtype(return_X_y=True)]


#2. 모델구성
model = RandomForestRegressor()


# Loop through the datasets and models, fit the models, and print the results
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) 

for i, dataset in enumerate(datasets):
    x, y = dataset
    print(f"\nResults for dataset {i+1}:")
    scores = cross_val_score(model, x, y, cv=kfold)
    print('acc:', scores, '\ncross_val_score 평균:', round(np.mean(scores),4))


'''
Results for dataset 1:
acc: [0.99887917 0.97730851 0.90285959 0.93898801 0.94936881]
cross_val_score 평균: 0.9535

Results for dataset 2:
acc: [0.85842044 0.90178154 0.78610344 0.87547003 0.80988722]
cross_val_score 평균: 0.8463

Results for dataset 3:
acc: [0.87789524 0.96934994 0.90817101 0.94826305 0.96582016]
cross_val_score 평균: 0.9339

Results for dataset 4:
acc: [0.82493868 0.90340586 0.8537513  0.86640928 0.84549338]
cross_val_score 평균: 0.8588

Results for dataset 5:
'''



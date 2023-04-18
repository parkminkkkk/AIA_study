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
models = [RandomForestRegressor(),DecisionTreeRegressor(),
          LogisticRegression(),LinearSVC()]


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) #n_splits=5 : 100%/5이므로 20%씩 #shuffle : 데이터 섞은 다음 20%씩 
# kfold = KFold() #데이터를 훈련시키는 위치에 따라서 결과값의 차이가 크게 날 수 있음

# Loop through the datasets and models, fit the models, and print the results
for i, dataset in enumerate(datasets):
    x, y = dataset
    print(f"\nResults for dataset {i+1}:")
    for j, model in enumerate(models):
        scores = cross_val_score(model, x, y, cv=kfold)
        print(f"Model {j+1}: {np.mean(scores):.3f}")


#acc: [1.         0.96666667 0.93333333 0.96666667 0.96666667]
#cross_val_score 평균: 0.9667


'''
Results for dataset 1:
Model 1: 0.953
Model 2: 0.926
Model 3: 0.973
Model 4: 0.967

Results for dataset 2:
Model 1: 0.842
Model 2: 0.697
Model 3: 0.945
Model 4: 0.838

Results for dataset 3:
Model 1: 0.934
Model 2: 0.875
Model 3: 0.938
Model 4: 0.860

Results for dataset 4:
Model 1: 0.862
Model 2: 0.671
Model 3: 0.962
Model 4: 0.945

Results for dataset 5:
Model 1: 0.927
Model 2: 0.838
Model 3: 0.620
'''
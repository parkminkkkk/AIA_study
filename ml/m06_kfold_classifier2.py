# all_estimators : 모든 모델에 대한 평가 (분류 41개 모델)

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.datasets import fetch_california_housing
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
datasets = [load_iris(return_X_y=True),
            load_breast_cancer(return_X_y=True), 
            load_wine(return_X_y=True),
            load_digits(return_X_y=True)]
dataname = ['iris', 'cancer', 'wine', 'digits']

scaler = [MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler]
scalername = ['MaxAbs', 'MinMax', 'Robus', 'Standard']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

###
best_models = {}  # 최고 모델 저장할 딕셔너리
#1. 데이터 
for i, v in enumerate(datasets):
    x, y = v
    for j in scaler:
        j=j()
        scalername=type(j).__name__
        x_scaled = j.fit_transform(x)
    
#2. 모델구성
        allAlgorithms = all_estimators(type_filter='classifier')
        max_score = 0
        max_name = 'max_model'
        
        for (name, algorithm) in allAlgorithms:
            try: #예외처리
                model = algorithm()

                scores = cross_val_score(model, x_scaled, y, cv=kfold)
                mean = round(np.mean(scores),4)
                # print('acc:', scores, '\ncross_val_score 평균:', mean)

                if max_score < mean: 
                    max_score = mean
                    max_name = name
            except:
                continue #continue: error 무시하고 계속 for문 돌리기 #break = for문 중단해라

        #dataset name , 최고모델, 성능
        print('========', dataname[i],'========')   
        print('best_scaler:', scalername)     
        print('최고모델:', max_name, max_score)
        print('================================')  
        # 최고 모델 딕셔너리에 저장
        best_models[(dataname[i], scalername)] = (max_name, max_score)

# 최고 모델 출력
print("========== 최고 모델 ==========")
for key, value in best_models.items():
    print(key, ":", value)

'''
======== iris ========
best_scaler: MaxAbsScaler
최고모델: LinearDiscriminantAnalysis 0.98
================================
======== iris ========
best_scaler: MinMaxScaler
최고모델: LinearDiscriminantAnalysis 0.98
================================
======== iris ========
best_scaler: RobustScaler
최고모델: LinearDiscriminantAnalysis 0.98
================================
======== iris ========
best_scaler: StandardScaler
최고모델: LinearDiscriminantAnalysis 0.98
================================
======== cancer ========
best_scaler: MaxAbsScaler
최고모델: SVC 0.9789
================================
======== cancer ========
best_scaler: MinMaxScaler
최고모델: SVC 0.9807
================================
======== cancer ========
best_scaler: RobustScaler
최고모델: LogisticRegression 0.9824
================================
======== cancer ========
best_scaler: StandardScaler
최고모델: SVC 0.9789
================================
======== wine ========
best_scaler: MaxAbsScaler
최고모델: RidgeClassifierCV 1.0
================================
======== wine ========
best_scaler: MinMaxScaler
최고모델: RidgeClassifierCV 1.0
================================
======== wine ========
best_scaler: RobustScaler
최고모델: RidgeClassifier 0.9944
================================
======== wine ========
best_scaler: StandardScaler
최고모델: RidgeClassifier 0.9944
================================
======== digits ========
best_scaler: MaxAbsScaler
최고모델: LabelPropagation 0.9878
================================
======== digits ========
best_scaler: MinMaxScaler
최고모델: LabelPropagation 0.9878
================================
======== digits ========
best_scaler: RobustScaler
최고모델: ExtraTreesClassifier 0.98
================================
'''



'''
======== iris ========
최고모델: LinearDiscriminantAnalysis 0.98
================================
======== cancer ========
최고모델: AdaBoostClassifier 0.9666
================================
======== wine ========
최고모델: RidgeClassifier 1.0
================================
======== digits ========
최고모델: SVC 0.9878
================================
'''











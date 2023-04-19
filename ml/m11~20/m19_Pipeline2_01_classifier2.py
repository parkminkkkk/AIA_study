#분류모델 - pipline사용
#삼중for문 : 1. 데이터셋, 2.스케일러, 3.모델 
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline



# 1. 데이터 
datasets = [(load_iris(return_X_y=True), 'Iris'), 
            (load_breast_cancer(return_X_y=True), 'Breast Cancer'),
            (load_wine(return_X_y=True), 'Wine'),
            (load_digits(return_X_y=True), 'Digits')]
dataname = ['iris', 'cancer', 'wine', 'digits']


# 2. 모델구성
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
models = [RandomForestClassifier(), DecisionTreeClassifier(), SVC()]

# max_score = 0

for data, data_name in datasets: 
    x, y = data
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337, stratify=y)
    # print(f'Data: {data_name}')
    max_score = 0

    for scaler in scalers:
        for model in models:
            pipeline = Pipeline([('s', scaler), ('m', model)])
            pipeline.fit(x_train, y_train)            
            score = pipeline.score(x_test, y_test)
            # print(f'{scaler.__class__.__name__} + {model.__class__.__name__} Score: {score:.4f}')

            y_pred = pipeline.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            # print(f'{scaler.__class__.__name__} + {model.__class__.__name__} Accuracy: {acc:.4f}')
            if max_score < acc:
                max_score = acc
                max_name = f'{scaler.__class__.__name__} + {model.__class__.__name__}'
    print('\n')
        #dataset name , 최고모델, 성능
    print('========', data_name,'========')        
    print('최고모델:', max_name, max_score)
    print('================================')  


'''
#Pipeline([('s', scaler), ('m', model)])
======== Iris ========
최고모델: MinMaxScaler + RandomForestClassifier 0.9666666666666667
================================

======== Breast Cancer ========
최고모델: RobustScaler + RandomForestClassifier 0.956140350877193
================================

======== Wine ========
최고모델: MinMaxScaler + RandomForestClassifier 1.0
================================

======== Digits ========
최고모델: MinMaxScaler + SVC 0.9861111111111112
================================
'''

'''
#make_pipeline(scaler, model)
======== Iris ========
최고모델: MaxAbsScaler + SVC 1.0
================================

======== Breast Cancer ========
최고모델: MinMaxScaler + SVC 0.9736842105263158
================================

======== Wine ========
최고모델: MinMaxScaler + RandomForestClassifier 0.9722222222222222
================================

======== Digits ========
최고모델: MinMaxScaler + SVC 0.975
================================
'''
        

'''
#make_pipeline(scaler, model)
#stratify=y
======== Iris ========
최고모델: MinMaxScaler + RandomForestClassifier 0.9666666666666667
================================


======== Breast Cancer ========
최고모델: MinMaxScaler + RandomForestClassifier 0.9649122807017544
================================


======== Wine ========
최고모델: MinMaxScaler + RandomForestClassifier 1.0
================================


======== Digits ========
최고모델: MinMaxScaler + SVC 0.9861111111111112
================================
'''
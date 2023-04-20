#분류모델 - pipline사용
#삼중for문 : 1. 데이터셋, 2.스케일러, 3.모델 
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
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
parameters = [
    {'randomforestclassifier__n_estimators' : [100,200], 'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10], 'randomforestclassifier__min_samples_split' : [2,3,5,10]},
    {'randomforestclassifier__min_samples_split' : [2,3,5,10]}]

#2. 모델구성
# pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])   
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe_model = [GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1), 
          RandomizedSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1), 
          HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1), 
          HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)]
modelsname = ['그리드서치', '랜덤서치', '할빙그리드서치', '할빙랜덤서치']


# max_score = 0

for data, data_name in datasets: 
    x, y = data
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337, stratify=y)
    # print(f'Data: {data_name}')
    for j,v in enumerate(pipe_model):
        max_score = 0
        models=v
        models.fit(x_train, y_train)
        y_predict = models.predict(x_test)
        acc= accuracy_score(y_test, y_predict)
        print("accuracy_score:", accuracy_score(y_test, y_predict))
        if max_score< acc:
            max_score = acc
            best_modelsname = modelsname[j]
    print('\n')
        #dataset name , 최고모델, 성능
    print('========', data_name,'========')        
    print('최고모델:', best_modelsname, max_score)
    print('================================')  


#그리드서치4가지 for문
'''
======== Iris ========
최고모델: 할빙랜덤서치 0.9666666666666667
================================
======== Breast Cancer ========
최고모델: 할빙랜덤서치 0.9385964912280702
================================
======== Wine ========
최고모델: 할빙랜덤서치 0.9722222222222222
================================
======== Digits ========
최고모델: 할빙랜덤서치 0.9694444444444444
================================
'''
#
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
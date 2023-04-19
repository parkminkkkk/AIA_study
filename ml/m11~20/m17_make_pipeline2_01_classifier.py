#분류모델 - pipline사용
#삼중for문 : 1. 데이터셋, 2.스케일러, 3.모델 
#[결과] : 각 데이터셋-스케일러-모델-성능의 최고만 출력
#ex) iris-minmax-RF-1.0

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


#1. 데이터 
dataset = [load_iris(return_X_y=True),load_breast_cancer(return_X_y=True),load_wine(return_X_y=True),
            load_digits(return_X_y=True)]

datasets_name = ['아이리스','캔서','와인','디짓']

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=337
# )

#2. 모델구성
scalers = [MinMaxScaler(), StandardScaler(),RobustScaler(), MaxAbsScaler()]
scaler_name = ['MINMAX','MAXABS','STANDARD','ROBUST']
models = [RandomForestClassifier(),DecisionTreeClassifier(),SVC()]
models_name = ['RandomForestClassifier','DecisionTreeClassifier','SVC']


for i , v in enumerate(dataset):
    x,y = v 
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    max_score = 0  # 데이터 쪽에서 종속시켜야됨
    max_name = 'd' # 데이터 쪽에서 종속시켜야됨
    scal_name = 'f' # 데이터 쪽에서 종속시켜야됨  그래야 아래서부터 진행됨
    for j, v1 in enumerate(scalers):
        for k,v2 in enumerate(models):
            model = make_pipeline(v1,v2)
            model.fit(x_train,y_train)
            #4.평가 예측
            result = model.score(x_test,y_test)
            # print('model.score :',result)
            y_predict = model.predict(x_test)
            acc = accuracy_score(y_test, y_predict)
            # print("accuracy_score:", acc)
            if max_score < result:
                max_score = result
                max_name = models_name[k]
                scal_name = scaler_name[j] 
    print('====================================')
    print(datasets_name[i],'최고 모델 :', max_name ,'최고 스케일러 :' ,scal_name, '\nmax_score', max_score)

'''
====================================
아이리스 최고 모델 : SVC 최고 스케일러 : ROBUST
max_score 1.0
====================================
캔서 최고 모델 : SVC 최고 스케일러 : MINMAX
max_score 0.9736842105263158
====================================
와인 최고 모델 : RandomForestClassifier 최고 스케일러 : MINMAX
max_score 0.9722222222222222
====================================
디짓 최고 모델 : SVC 최고 스케일러 : MINMAX
max_score 0.975
'''
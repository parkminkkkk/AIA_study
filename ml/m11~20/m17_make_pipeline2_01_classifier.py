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


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=337
# )

#2. 모델구성
scaler = [MinMaxScaler(), StandardScaler(),RobustScaler(), MaxAbsScaler()]
models = [RandomForestClassifier(),DecisionTreeClassifier(),SVC()]

# model = make_pipeline(StandardScaler(), SVC())


for i in dataset: 
    x, y = i
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    best_accuracy = 0
    best_model = None
    best_scaler = None
    best_data = None
    for j in scaler:
        for k in models:
            model = make_pipeline(j,k)
            model.fit(x_train,y_train)            
            result = model.score(x_test, y_test)
            # print("model.score:", result)
            y_predict = model.predict(x_test)
            acc = accuracy_score(y_test, y_predict)
            # print("accuracy_score:", acc)
            print(f"Scaler: {j}, Model: {k}, Accuracy Score: {acc}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = k
                best_scaler = j
                best_data = i
    print('========', dataset[i],'========')        
    print('최고모델:', best_data, best_model, best_scaler)
    print('================================')  

#[과제/실습]
# 피처를 한개씩 삭제하고 성능비교
# 10개의 데이터셋, 파일 생성 / 피처를 한개씩 삭제하고 성능비교 
# 모델은 RF로만 한다. 


from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

data_list = [load_iris, load_breast_cancer, load_wine, load_digits, fetch_california_housing, load_diabetes]
data_list_name = ['load_iris', 'load_breast_cancer', 'load_wine', 'load_digits', 'fetch_california_housing', 'load_diabetes']
for i in range(len(data_list)):
    # 1. 데이터
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if i<4:
        # 2. 모델
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    result = model.score(x_test, y_test)
    print(data_list_name[i], 'model.score : ', result)

    y_pred = model.predict(x_test)
    if i<4:
        acc = accuracy_score(y_test, y_pred)
        print(data_list_name[i], 'acc : ', acc)

        a = model.feature_importances_
        a = a.argmin(axis=0)

        x_d = pd.DataFrame(x).drop([a], axis=1) 
        x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(x_d, y, train_size=0.8, shuffle=True, random_state=337)
        scaler = MinMaxScaler()
        x_train_d = scaler.fit_transform(x_train_d)
        x_test_d = scaler.transform(x_test_d)

        # 2. 모델
        model = RandomForestClassifier()

        # 3. 훈련
        model.fit(x_train_d, y_train_d)

        # 4. 평가, 예측
        result = model.score(x_test_d, y_test_d)
        print(data_list_name[i], f'drop {a}th col score : ', result)

        y_pred_d = model.predict(x_test_d)
        acc = accuracy_score(y_test_d, y_pred_d)
        print(data_list_name[i], f'drop {a}th col acc : ', acc, '\n')
    else:
        r2 = r2_score(y_test, y_pred)
        print(data_list_name[i], 'r2 : ', r2)

        a = model.feature_importances_
        a = a.argmin(axis=0)

        x_d = pd.DataFrame(x).drop([a], axis=1) 
        x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(x_d, y, train_size=0.8, shuffle=True, random_state=337)
        scaler = MinMaxScaler()
        x_train_d = scaler.fit_transform(x_train_d)
        x_test_d = scaler.transform(x_test_d)

        # 2. 모델
        model = RandomForestRegressor()

        # 3. 훈련
        model.fit(x_train_d, y_train_d)

        # 4. 평가, 예측
        result = model.score(x_test_d, y_test_d)
        print(data_list_name[i], f'drop {a}th col score : ', result)

        y_pred_d = model.predict(x_test_d)
        r2 = r2_score(y_test_d, y_pred_d)
        print(data_list_name[i], f'drop {a}th col r2 : ', r2, '\n')
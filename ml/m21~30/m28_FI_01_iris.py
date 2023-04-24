# [실습]
# feature_importance가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성 후 모델 돌려서 결과 도출
# 기존 모델들과 성능 비교 
# 모델 4개 구성

#결과비교 (삭제전-삭제후)
'''
ex) 
1. DT
#기존acc : 
#컬럼 삭제후 acc: 
2. RF
3. GDB
4. XGB
'''
####################################################
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV   
from sklearn.metrics import accuracy_score


#1. 데이터 
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)

#2. 모델
model_list = [RandomForestClassifier(),
GradientBoostingClassifier(),
DecisionTreeClassifier(),
XGBClassifier()]

# 3. 훈련
for model, value in enumerate(model_list) :
    model = value
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    
    print("accuracy_score : ", acc)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    x_drop = pd.DataFrame(x).drop(idx, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train, y_train)

    result = model.score(x_test, y_test)
    print("acc after feature selection: ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)

    print("accuracy_score after feature selection: ", acc)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
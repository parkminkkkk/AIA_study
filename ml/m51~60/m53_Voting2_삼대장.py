import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, #stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingRegressor 
from xgboost import XGBRegressor             #데이터 만개이상일떄 좋음 max_depth 디폴트8
from lightgbm import LGBMRegressor           #max_depth 디폴트 30 / leaf_node 차이(xgb보다 더 깊이 들어감, 더 좋은 곳의 트리를 선택해 그곳에서 더 깊이 연산)
from catboost import CatBoostRegressor       #정의, 차이점(파라미터) 



xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

model = VotingRegressor(estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],   #보팅 평가자 모델
                        #   voting='soft',  #디폴트 'hard'    
                        # *Regressor : voting안먹힘 => regressor는 voting을 선택할 수 없으므로 평균내서 함
                        #  (배깅과의 차이점 : 배깅(한가지모델), 보팅(여러 모델))

                          )              


#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
print("model.score:", model.score(x_test, y_test))
print("Voting R2:", r2_score(y_test, y_pred))

regressors = [xgb, lg, cat]
for model2 in regressors :
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score2))


'''
model.score: 0.8529607932472343
Voting R2: 0.8529607932472343
XGBRegressor 정확도 : 0.8331
LGBMRegressor 정확도 : 0.8413
CatBoostRegressor 정확도 : 0.8571
'''
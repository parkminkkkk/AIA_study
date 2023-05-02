import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import VotingRegressor 
from xgboost import XGBRegressor            
from lightgbm import LGBMRegressor          
from catboost import CatBoostRegressor       

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, #stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

model = VotingRegressor(estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],   #보팅 평가자 모델
                        #   voting='soft',  #디폴트 'hard'

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
#배깅
model.score: 0.47835953603852366
r2: 0.47835953603852366

#보팅
model.score: 0.5345297780787086
Voting R2: 0.5345297780787086  
XGBRegressor 정확도 : 0.4619
LGBMRegressor 정확도 : 0.5229
CatBoostRegressor 정확도 : 0.5380
'''
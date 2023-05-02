#파일 각각 만들어서 비교 
#iris, cancer, dacon_diabets, wine, fetch_covtpe, digits
#diabets, california, dacon_ddarung, kaggle_bike

import numpy as np 
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
path = 'd:/study/_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
# print(train_csv.isnull().sum()) 
#결측치 없음

###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, #stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.ensemble import VotingRegressor, StackingRegressor
from xgboost import XGBRegressor            
from lightgbm import LGBMRegressor          
from catboost import CatBoostRegressor    
xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

model = StackingRegressor(estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],   #보팅 평가자 모델
                        #   voting='soft',  #디폴트 'hard'

                          )              


#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
print("model.score:", model.score(x_test, y_test))
print("Stacking R2:", r2_score(y_test, y_pred))

regressors = [xgb, lg, cat]
for model2 in regressors :
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score2))


'''
#stacking
model.score: 0.3448407243418167
Stacking R2: 0.3448407243418167
XGBRegressor 정확도 : 0.3094
LGBMRegressor 정확도 : 0.3365
CatBoostRegressor 정확도 : 0.3386

#배깅
model.score: 0.20435180115285279
acc: 0.20435180115285279

#보팅
model.score: 0.3414482207343491
Voting R2: 0.3414482207343491  
XGBRegressor 정확도 : 0.3094
LGBMRegressor 정확도 : 0.3365
CatBoostRegressor 정확도 : 0.3386
'''
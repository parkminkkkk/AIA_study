# all_estimators : 모든 모델에 대한 평가

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators

import sklearn as sk
print(sk.__version__)    #1.0.2
import warnings
warnings.filterwarnings('ignore')



#1. 데이터 
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성 
# model = RandomForestRegressor(n_estimators=120, n_jobs=4) #n_estimators :트리 개수 = epochs/  #n_jobs : cpu사용 개수(4개 모두사용하겠다) 
allAlgorithms = all_estimators(type_filter='regressor')    #회귀모델
# allAlgorithms = all_estimators(type_filter='classifier')  #분류모델

print('allAlgorithms:', allAlgorithms)
print('모델의 개수 :', len(allAlgorithms)) #55



'''
#3. 컴파일, 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측 
r2 = model.score(x_test, y_test)
print("model_score: ", r2)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print("r2_score: ", r2_score)
'''

'''
model_score:  0.8150037720889283
r2_score:  0.8150037720889283
'''
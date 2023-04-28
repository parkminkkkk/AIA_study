# [실습] smote 적용

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgbm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score


#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)
print(np.unique(y, return_counts=True)) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, #stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성 
# model = XGBClassifier()
model = RandomForestClassifier(random_state=3377)

#3. 컴파일, 훈련 
model.fit(x_train, y_train)  

  
#4. 평가예측 
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print("==============SMOTE 적용 전=======================")
print("최종점수 :", results)
print("acc:", acc)
print("f1(macro):", f1)

smote = SMOTE(random_state=321,k_neighbors=2) #k_neighbors 디폴트5
x_train, y_train= smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape) 
print(np.unique(y_train, return_counts=True)) 

#2. 모델구성 
# model = XGBClassifier()
model = RandomForestClassifier(random_state=3377)

#3. 컴파일, 훈련 
model.fit(x_train, y_train)  

  
#4. 평가예측 
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print("==============SMOTE 적용 후=======================")
print("최종점수 :", results)
print("acc:", acc)
print("f1(macro):", f1)
# [실습] smote 적용

import numpy as np
import pandas as pd
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
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]
print(train_csv.shape) #(5497,13)
 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)   #[1 0 1 ... 1 1 1]
print(type(aaa))  #<class 'numpy.ndarray'>
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white'])) #[0 1]


#1-1 결측치 제거 
# print(train_csv.isnull().sum()) #결측치없음 

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']
# y = y-3
print(type(y))
print(y)
print("y_shape:", y.shape)           #(5497,)
print('y의 라벨값 :', np.unique(y, return_counts=True))  #[3 4 5 6 7 8 9]
# y의 라벨값 : (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
# test_csv = test_csv.drop(['type'], axis=1)

#1-3 데이터분리 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=640874, stratify=y)

#1-4 스케일링 
# scaler = MinMaxScaler() 
scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
# print(np.min(x_test), np.max(x_test)) 

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
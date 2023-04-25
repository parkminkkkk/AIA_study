#RFC디폴트 값 : RFC_LDA값 비교 

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_diabetes, fetch_covtype
from sklearn.metrics import accuracy_score
# 1. 데이터
x, y = load_iris(return_X_y=True) 
print(x.shape) #(150,4)

lda = LinearDiscriminantAnalysis()
x = lda.fit_transform(x,y) 
print(x.shape) #(150, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=640, train_size=0.8, shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성 
model = RandomForestClassifier(random_state=640)

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가, 예측 
result = model.score(x_test, y_test)
print('result:', result)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print('acc:', acc)

'''
#기본(LDA전)
result: 0.9333333333333333

#LDA
result: 0.9666666666666667
'''

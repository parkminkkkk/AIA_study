import numpy as np 
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

#1. 데이터
data_list = [load_diabetes(), fetch_california_housing()]
data_name = ['diabetes', 'callifornia']

for i in range(len(data_list)):
    data = data_list[i]
    x, y = data.data, data.target
    pf = PolynomialFeatures()
    x_pf = pf.fit_transform(x)
    print("=======", data_name[i], "========")
    print(x.shape, "=>", x_pf.shape)

    x_train, x_test, y_train, y_test= train_test_split(
        x, y, train_size=0.8, random_state=3377)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #모델구성
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    #4. 평가
    y_pred = model.predict(x_test)
    print("model.score:", model.score(x_test, y_test))
    print("r2:", r2_score(y_test, y_pred))


'''
======= diabetes ========
(442, 10) => (442, 66)
model.score: 0.3288117554890846
r2: 0.3288117554890846
======= callifornia ========
(20640, 8) => (20640, 45)
model.score: 0.7947539939983158
r2: 0.7947539939983158
'''



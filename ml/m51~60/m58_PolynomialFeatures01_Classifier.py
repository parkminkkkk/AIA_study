import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

#1. 데이터
data_list = [load_iris(), load_breast_cancer(), load_wine(), load_digits()]
data_name = ['iris', 'cancer', 'wine', 'digits']

for i in range(len(data_list)):
    data = data_list[i]
    x, y = data.data, data.target
    pf = PolynomialFeatures()
    x_pf = pf.fit_transform(x)
    print("=======", data_name[i], "========")
    print(x.shape, "=>", x_pf.shape)

    x_train, x_test, y_train, y_test= train_test_split(
        x, y, train_size=0.8, random_state=3377, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #모델구성
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    #4. 평가
    y_pred = model.predict(x_test)
    print("model.score:", model.score(x_test, y_test))
    print("ACC:", accuracy_score(y_test, y_pred))






'''
======= iris ========
(150, 4) => (150, 15)
model.score: 0.9666666666666667
ACC: 0.9666666666666667
======= cancer ========
(569, 30) => (569, 496)
model.score: 0.9912280701754386
ACC: 0.9912280701754386
======= wine ========
(178, 13) => (178, 105)
model.score: 0.9722222222222222
ACC: 0.9722222222222222
======= digits ========
(1797, 64) => (1797, 2145)
model.score: 0.9861111111111112
ACC: 0.9861111111111112
'''


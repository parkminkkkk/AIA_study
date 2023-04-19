#feature_importances 
#트리계열에서만 제공함!! model.feature_importances_
#컬럼의 종류에 따라 훈련결과에 악영향을 주는 불필요한 컬럼 존재함 (노이즈)
#=> 컬럼(열, 특성, feature) 걸러내는 작업 필요함 

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

#1. 데이터 
datalist = [load_iris(return_X_y=True), load_breast_cancer(return_X_y=True), load_wine(return_X_y=True)]
model = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]
modelname = ['DTC', 'RFC', 'GBC', 'XGB']


for j in datalist:
    x,y= j
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    for i, v in enumerate(model):
        v.fit(x_train, y_train)
        result = v.score(x_test, y_test)
        y_predict = v.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict, average='macro')
        # print(modelname[i], ":", "acc:", accuracy_score)
        print(modelname[i], ":", "f1:", f1_score)
        print(modelname[i], ":", "컬럼별 중요도", v.feature_importances_)
        print('-------------------------------------------')

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
datalist = [load_iris, load_breast_cancer, load_wine]
models = [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier]
modelname = ['DTC', 'RFC', 'GBC', 'XGB']


for j in datalist:
    x,y= j(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    for v in models:
        model = v()
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        # f1 = f1_score(y_test, y_predict, average='macro')
        print(type(model).__name__, ":", "acc:", accuracy_score)
        # print(modelname[i], ":", "f1:", f1_score)
        print(type(model).__name__, ":", "컬럼별 중요도", model.feature_importances_)
        print('-------------------------------------------')

'''
-------------------------------------------
DecisionTreeClassifier : acc: <function accuracy_score at 0x000001FA1CD11A60>
DecisionTreeClassifier : 컬럼별 중요도 [0.         0.         0.01029857 0.01713682 0.         0.01071051
 0.16522355 0.         0.03060147 0.30744166 0.03998592 0.
 0.41860149]
-------------------------------------------
RandomForestClassifier : acc: <function accuracy_score at 0x000001FA1CD11A60>
RandomForestClassifier : 컬럼별 중요도 [0.14047264 0.03616757 0.01609947 0.02035893 0.02615247 0.04892502
 0.20443341 0.01569231 0.01790476 0.15846149 0.08857061 0.08608507
 0.14067624]
-------------------------------------------
GradientBoostingClassifier : acc: <function accuracy_score at 0x000001FA1CD11A60>
GradientBoostingClassifier : 컬럼별 중요도 [2.07766444e-03 5.63123863e-02 1.41744416e-02 1.74961391e-03
 4.04893118e-03 1.80109800e-04 2.99262752e-01 5.06255575e-04
 7.94833690e-04 2.46315065e-01 6.13587717e-02 1.48760580e-02
 2.98343118e-01]
-------------------------------------------
XGBClassifier : acc: <function accuracy_score at 0x000001FA1CD11A60>
XGBClassifier : 컬럼별 중요도 [0.03312182 0.12021569 0.01665129 0.         0.03627352 0.
 0.24492191 0.0207895  0.01708013 0.20495585 0.06092481 0.01492534
 0.23014012]
-------------------------------------------
'''
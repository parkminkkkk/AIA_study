import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터 
data_list = [load_iris(return_X_y=True), load_breast_cancer(return_X_y=True), load_wine(return_X_y=True)]
model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
dtname_list = ['iris:', 'cancer:', 'wine:']
mdname_list = ['LinearSVC:', 'LogisticRegression:', 'DecisionTreeClassifier:', 'RandomForestClassifier:']


#2. 모델
for i,v in enumerate(data_list):  #순서(i), 값(v) 반환
    x, y = v
    # print(x.shape, y.shape)
    print("===================================")
    print(dtname_list[i])
    for j,v2 in enumerate(model_list):
        model = v2
        #컴파일, 훈련
        model.fit(x,y)
        #평가, 예측 
        results = model.score(x,y)
        print(mdname_list[j], 'model.score:',results)
        y_predict = model.predict(x)
        acc = accuracy_score(y,y_predict)
        print(mdname_list[j], 'accuracy_score:',acc)

'''
iris:
LinearSVC: 0.9666666666666667
LogisticRegression: 0.9733333333333334
DecisionTreeRegressor: 1.0
RandomForestRegressor: 0.99262
===================================
cancer:
LinearSVC: 0.9314586994727593
LogisticRegression: 0.9472759226713533
DecisionTreeRegressor: 1.0
RandomForestRegressor: 0.9798101382062259
===================================
wine:
LinearSVC: 0.9044943820224719
LogisticRegression: 0.9662921348314607
DecisionTreeRegressor: 1.0
RandomForestRegressor: 0.9916149537648613
'''
'''
(150, 4) (150,)
0.9666666666666667
(569, 30) (569,)
0.5817223198594025
(178, 13) (178,)
0.8202247191011236
'''
'''
(150, 4) (150,)
0.9666666666666667
0.9733333333333334
1.0
0.992188
(569, 30) (569,)
0.8822495606326889
0.9472759226713533
1.0
0.979103436657682
(178, 13) (178,)
0.6797752808988764
0.9662921348314607
1.0
0.9920927556142668
'''




#pickle : 가중치 load

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#########################################
#2. 모델 - 피클 불러오기 
import pickle
path = './_save/pickle_test/'
model = pickle.load(open(path + 'm43_pickle1_save.dat', 'rb'))


#4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 :", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc:", acc)

# 최종 점수 : 0.9736842105263158
# acc: 0.9736842105263158
# score = acc임 (동일)

#########################################
# import pickle
# path = './_save/pickle_test/'
# pickle.dump(model, open(path + 'm43_pickle1_save.dat', 'wb'))


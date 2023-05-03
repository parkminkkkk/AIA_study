# stacking : 각각의 모델을 훈련시켜 predict만들어냄 -> 이후 predict을 마지막 모델에서 다시 훈련시켜서 결과 에측함
# 이때, 마지막 모델에서 훈련시키는 데이터가 test데이터이므로 과적합의 문제를 일으킬 수 있음  **과적합주의**  / =>train데이터를 두개로 나누어서 사용했음


import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import VotingRegressor 
from xgboost import XGBRegressor, XGBClassifier           
from lightgbm import LGBMRegressor, LGBMClassifier          
from catboost import CatBoostRegressor, CatBoostClassifier 

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)

li = []
models = [xgb, lg, cat]
for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # print(y_predict.shape)  #(114,) : 벡터-> 열 형태로 변화해줘야함
    y_predict = y_predict.reshape(y_predict.shape[0], 1)  
    li.append(y_predict)

    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print("{0}ACC: {1:.4f}".format(class_name, score))

# print(li) #numpy형태로 들어가있음 (벡터형태) / list형태로는 훈련 못시키므로 list안에 있는 np들을 concat해주면 됨 
y_stacking_pred = np.concatenate(li, axis=1)
# print(y_stacking_pred.shape)  #(342,)  #(-1,3)이 나와야함...(3개를 이어붙였으니까.. ) //  ##최종 => (114, 3) 
#====================================================================#
# axis=1일때 벡터형태로 되어있음 (error :  행렬형태로 줘야함!)
# numpy.AxisError: axis 1 is out of bounds for array of dimension 1
#====================================================================#

model = CatBoostClassifier(verbose=0)
model.fit(y_stacking_pred, y_test)                #스테킹 과적합 주의 : 
score = model.score(y_stacking_pred, y_test)
print("스태킹 결과:", score)


'''
#스테킹 
XGBClassifierACC: 0.9912
LGBMClassifierACC: 0.9825
CatBoostClassifierACC: 0.9912
스태킹 결과: 0.9912280701754386
'''


import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
aaa = LogisticRegression()
model = BaggingClassifier(aaa,      #배깅에 쓰는 모델 / 모델 훈련 10번
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True,            #통상 True/ #샘플의 중복을 허용 (디폴트=True)
                          )              



#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
print("model.score:", model.score(x_test, y_test))
print("acc:", accuracy_score(y_test, y_pred))

'''
#DTC
model.score: 0.9473684210526315
acc: 0.9473684210526315

#RF
model.score: 0.9649122807017544
acc: 0.9649122807017544

#배깅-DTC10번 / bootstrap=True(디폴트)  #샘플의 중복을 허용
model.score: 0.9912280701754386
acc: 0.9912280701754386

#배깅-DTC10번/ bootstrap=False
model.score: 0.956140350877193
acc: 0.956140350877193

#배깅- LogisticRegression
model.score: 0.9736842105263158
acc: 0.9736842105263158
'''
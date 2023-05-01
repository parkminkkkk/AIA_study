#파일 각각 만들어서 비교 
#iris, cancer, dacon_diabets, wine, fetch_covtpe, digits
#diabets, california, dacon_ddarung, kaggle_bike

import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),  #배깅에 쓰는 모델 DTC/ 모델 훈련 10번
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

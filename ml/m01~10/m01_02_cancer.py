import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

#2. 모델구성
# model = RandomForestRegressor()
model = RandomForestClassifier()


#3. 컴파일, 훈련 
model.fit(x,y)

#4. 평가, 예측 
results= model.score(x,y) #evaluate없음. score
print(results)

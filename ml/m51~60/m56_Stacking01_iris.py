# stacking : 각각의 모델을 훈련시켜 predict만들어냄 -> 이후 predict을 마지막 모델에서 다시 훈련시켜서 결과 에측함
# 이때, 마지막 모델에서 훈련시키는 데이터가 test데이터이므로 과적합의 문제를 일으킬 수 있음  **과적합주의**  / =>train데이터를 두개로 나누어서 사용했음


import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()

#스테킹안에 배깅, 보팅 가능 
#배깅, 보팅한 것 스테킹 가능
model = StackingClassifier(estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],  
                        #    final_estimator=LogisticRegression(),              #마지막 평가 모델
                        #    final_estimator=RandomForestClassifier(),            
                        #    final_estimator=VotingClassifier(estimators=[('LR', lr), ('KNN', knn), ('DT', dt)])   #스테킹안에 보팅도 가능
                          )              

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
print("model.score:", model.score(x_test, y_test))
print("Stacking ACC:", accuracy_score(y_test, y_pred))

classifiers = [lr, knn, dt]
for model2 in classifiers :
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score2))

'''
#stacking
model.score: 0.9
Stacking ACC: 0.9
LogisticRegression 정확도 : 0.9333
KNeighborsClassifier 정확도 : 0.9333
DecisionTreeClassifier 정확도 : 0.8333
'''
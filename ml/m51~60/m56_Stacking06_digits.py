import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import StackingClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()

model = StackingClassifier( estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],   #보팅 평가자 모델
                            
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
#스테킹
model.score: 0.9888888888888889
Stacking ACC: 0.9888888888888889
LogisticRegression 정확도 : 0.9750
KNeighborsClassifier 정확도 : 0.9778
DecisionTreeClassifier 정확도 : 0.8639

#배깅
model.score: 0.9472222222222222
acc: 0.9472222222222222

#보팅_hard
model.score: 0.9861111111111112
Voting ACC: 0.9861111111111112
LogisticRegression 정확도 : 0.9750
KNeighborsClassifier 정확도 : 0.9778
DecisionTreeClassifier 정확도 : 0.8833

#보팅_soft
model.score: 0.9916666666666667
Voting ACC: 0.9916666666666667
LogisticRegression 정확도 : 0.9750
KNeighborsClassifier 정확도 : 0.9778
DecisionTreeClassifier 정확도 : 0.8833
'''
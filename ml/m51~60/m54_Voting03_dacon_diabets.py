import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터 
path = 'd:/study/_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
test_csv= pd.read_csv(path+'test.csv', index_col=0)
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()

model = VotingClassifier( estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],   #보팅 평가자 모델
                          voting='hard',  #디폴트 'hard'
                        #   voting='soft'
                          )              


#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
print("model.score:", model.score(x_test, y_test))
print("Voting ACC:", accuracy_score(y_test, y_pred))

classifiers = [lr, knn, dt]
for model2 in classifiers :
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score2))

'''
#배깅
model.score: 0.6717557251908397
acc: 0.6717557251908397

#보팅_hard
model.score: 0.7251908396946565
Voting ACC: 0.7251908396946565
LogisticRegression 정확도 : 0.7634
KNeighborsClassifier 정확도 : 0.7023
DecisionTreeClassifier 정확도 : 0.6718

#보팅_soft
model.score: 0.6717557251908397
Voting ACC: 0.6717557251908397
LogisticRegression 정확도 : 0.7634
KNeighborsClassifier 정확도 : 0.7023
DecisionTreeClassifier 정확도 : 0.6870
'''
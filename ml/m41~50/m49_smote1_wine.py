#smote : 불균형 데이터 처리 방법 (증폭)

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터 
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x. shape, y.shape) #(178, 13) (178,)  #y : 178개의 스칼라가 모인 벡터형태.1차원 (np) = (pd) 시리즈형태 (2차원부터 데이터프레임)
print(np.unique(y, return_counts=True))  #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts().sort_index())
'''
0    59
1    71
2    48
dtype: int64
'''
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

'''
x = x[:-25]
y = y[:-25]
print(x.shape, y.shape) #(153, 13) (153,)
print(y)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 23], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=321,
    stratify=y
)
print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2]), array([44, 53, 17], dtype=int64))

#2. 모델 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=321)

#3. 훈련
model.fit(x_train, y_train)
#4. 평가 
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print("==============SMOTE 적용 전=======================")
print("model.score:", score)
print('accuarcy:',  accuracy_score(y_test, y_predict))
print('f1_score(macro)', f1_score(y_test, y_predict, average='macro')) #average= one of [None, 'micro', 'macro', 'weighted'].
# print('f1_score(micro)', f1_score(y_test, y_predict, average='micro')) #average= one of [None, 'micro', 'macro', 'weighted'].

#y의 클래스 개수가 한쪽으로 치우친 데이터의 경우 acc보다는 f1_score가 더 정확한 지표임
#원래 f1_score는 이진분류에 사용함(average='binary')/ 다중분류에서 사용할때에는 'macro', 'micro' (주로 'macro'사용) : 가중치의 평균을 계산하는 방식..
'''
model.score: 0.9487179487179487
accuarcy: 0.9487179487179487
f1_score(macro) 0.9439984430496765
f1_score(micro) 0.9487179487179487
'''
print("==============SMOTE 적용 후=======================")
#####smote적용시 모두 53개가 됨(최대 개수에 맞춰서 증폭시킴)####
#(array([0, 1, 2]), array([44, 53, 17], dtype=int64))

smote = SMOTE(random_state=321, k_neighbors=8) #k_neighbors 디폴트5
# k_neighbors 최근접 이웃방식 : (k개 가장 가까이 있는 이웃의 개수, 그 이웃들의 중심에 데이터가 생성, 따라서 데이터가 몰림(중복)가능성이 있다는 것이 단점)
x_train, y_train= smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape) #(159, 13) (159,)/3   = 53개 
print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2]), array([53, 53, 53], dtype=int64))
#통상 train데이터만 증폭시킴, test데이터는 건들이지 않음(왜냐하면 평가,예측할때 사용하는 것이므로 데이터 건들이지xx)

#2. 모델 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=321)

#3. 훈련
model.fit(x_train, y_train)
#4. 평가 
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print("model.score:", score)
print('accuarcy:',  accuracy_score(y_test, y_predict))
print('f1_score(macro)', f1_score(y_test, y_predict, average='macro'))



'''
==============SMOTE 적용 전=======================
model.score: 0.967741935483871
accuarcy: 0.967741935483871
f1_score(macro) 0.957351290684624
==============SMOTE 적용 후=======================
(171, 13) (171,)
(array([0, 1, 2]), array([57, 57, 57], dtype=int64))
model.score: 1.0
accuarcy: 1.0
f1_score(macro) 1.0
'''

#SMOTE의 단점 : 증폭하는데 시간이 오래 걸림 
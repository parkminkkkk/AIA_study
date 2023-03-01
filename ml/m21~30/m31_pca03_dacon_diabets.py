# PCA : 차원(컬럼) 축소(압축)
# target(Y)는 축소 안함 -> X컬럼들만 차원 축소시킴  
# 즉, 타겟값이 없음, 타겟값 생성함(비지도학습 unsupervised learning) : 스케일링 개념
#1. y값을 찾는 비지도학습
#2. 전처리개념 스케일링
# 컬럼 간의 좌표를 찍었을때, 그려지는 직선위로 데이터들의 좌표를 맵핑한다. 

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# [실습]
# for문으로 pca 10~1개까지 

#1. 데이터 
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'
train_csv= pd.read_csv(path+'train.csv', index_col=0)
test_csv= pd.read_csv(path+'test.csv', index_col=0)
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']



for i in range(8, 0, -1):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestClassifier(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")

'''
n_coponets=8,  결과: 0.7404580152671756 
n_coponets=7,  결과: 0.7480916030534351 
n_coponets=6,  결과: 0.7480916030534351 
n_coponets=5,  결과: 0.6946564885496184 
n_coponets=4,  결과: 0.6946564885496184 
n_coponets=3,  결과: 0.7175572519083969 
n_coponets=2,  결과: 0.6641221374045801 
n_coponets=1,  결과: 0.6335877862595419 
'''



# #1. 데이터 
# datasets = load_diabetes()
# print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# x = datasets['data']
# y = datasets.target
# print(x.shape, y.shape)    #(442, 10) (442,)

# #데이터x 컬럼 축소
# pca = PCA(n_components=7)
# x = pca.fit_transform(x)
# print(x.shape)             #(442, 5)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=123, shuffle=True,
# )

# #2. 모델구성 
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=123)

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가,예측
# results = model.score(x_test, y_test)
# print("결과:", results)

'''
#pca_before
결과: 0.5260875642282989

#n_components=7
결과: 0.5141328515687419
'''
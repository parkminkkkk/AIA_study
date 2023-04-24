# PCA : 차원(컬럼) 축소(압축)
# target(Y)는 축소 안함 -> X컬럼들만 차원 축소시킴  
# 즉, 타겟값이 없음, 타겟값 생성함(비지도학습 unsupervised learning) : 스케일링 개념
#1. y값을 찾는 비지도학습
#2. 전처리개념 스케일링
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# [실습]
# for문으로 pca 30~1개까지(3개단위로) => 기본결과, 3개축소,6개축소...30개축소

#1. 데이터 
datasets = load_breast_cancer()
print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)    #(442, 10) (442,)

for i in range(30, 0, -3):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestClassifier(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")

'''
n_coponets=30,  결과: 0.8699226862679585 
n_coponets=27,  결과: 0.884206014032743 
n_coponets=24,  결과: 0.8852496491814233 
n_coponets=21,  결과: 0.8846821249582358 
n_coponets=18,  결과: 0.8935949214834614 
n_coponets=15,  결과: 0.8884795856999665 
n_coponets=12,  결과: 0.8974609421984631 
n_coponets=9,  결과: 0.8892984964918142 
n_coponets=6,  결과: 0.8981198797193451 
n_coponets=3,  결과: 0.8009856331440026 
'''


'''
#1. 데이터 
datasets = load_breast_cancer()
# print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)    #(569, 30) (569,)

# #데이터x 컬럼 축소
# pca = PCA(n_components=7)
# x = pca.fit_transform(x)
# print(x.shape)             #(442, 5)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,
)

#2. 모델구성 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
results = model.score(x_test, y_test)
print("결과:", results)
'''


'''
결과: 0.9162159037754761
'''

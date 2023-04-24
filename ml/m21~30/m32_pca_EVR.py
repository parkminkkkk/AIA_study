# PCA : 차원(컬럼) 축소(압축)
# target(Y)는 축소 안함 -> X컬럼들만 차원 축소시킴  
# 즉, 타겟값이 없음, 타겟값 생성함(비지도학습 unsupervised learning) : 스케일링 개념
#1. y값을 찾는 비지도학습
#2. 전처리개념 스케일링
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


#1. 데이터 
datasets = load_breast_cancer()
# print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)    #(569, 30) (569,)

#데이터x 컬럼 축소
pca = PCA(n_components=7)
x = pca.fit_transform(x)
print(x.shape)             #(442, 5)


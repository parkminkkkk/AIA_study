# PCA : 차원(컬럼) 축소(압축)
# target(Y)는 축소 안함 -> X컬럼들만 차원 축소시킴  
# 즉, 타겟값이 없음, 타겟값 생성함(비지도학습 unsupervised learning) : 스케일링 개념
#1. y값을 찾는 비지도학습
#2. 전처리개념 스케일링

#np.cumsum(pca_EVR) 몇개정도로 줄일 것인지 판단 가능 (최대차원축소의 개수 참고가능)

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
print(x.shape, y.shape)    #(569, 10) (569,)

#데이터x 컬럼 축소
pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x.shape)             #(569, 30)

#설명가능한 변화율
pca_EVR = pca.explained_variance_ratio_  
print(pca_EVR)
'''
[9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
 8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
 3.44135279e-07 1.86018721e-07 6.99473205e-08 1.65908880e-08
 6.99641650e-09 4.78318306e-09 2.93549214e-09 1.41684927e-09
 8.29577731e-10 5.20405883e-10 4.08463983e-10 3.63313378e-10
 1.72849737e-10 1.27487508e-10 7.72682973e-11 6.28357718e-11
 3.57302295e-11 2.76396041e-11 8.14452259e-12 6.30211541e-12
 4.43666945e-12 1.55344680e-12]
'''
'''
pca1번째, 2번쨰, 3번쨰, 4번째,5번째,
.
.
. 
'''
print(sum(pca_EVR))       #0.9999999999999998

pca_cumsum = np.cumsum(pca_EVR)  #배열의 누적합
print(pca_cumsum)
'''
[0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
 0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
 0.99999999 0.99999999 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.        ]
'''
'''
pca1개, 2개(1번째+2번쨰), 3개(1+2+3번쨰), 4개, 5개,... 
(pca1개:원본과 일치률이 0.98정도)
.
.
.
pca15번쨰 =1. : 15번했을때까지는 데이터 손실이 없을 것이다. 
 #이를 보고, 몇개정도로 줄일 것인지 판단 가능 (최대차원축소의 개수 참고가능)
'''

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()
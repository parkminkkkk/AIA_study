# Linear Discriminant Analsis
# <LDA> 
# - 각 데이터의 클래스 별로 맵핑함(매치시킴)
# -실질적으로 지도학습(y의 값을 알아야 함)/ 차원축소에 많이 쓰임
# <PCA와의 차이점> 
# -PCA는 데이터의 방향성에 대해서 선을 그은 후 그에 따른 데이터들의 선에 맵핑한다(매칭시킴) 
# -비지도학습/ 차원축소에 많이 쓰임

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from tensorflow.keras.datasets import cifar100

# # 1. 데이터
# x, y = load_digits(return_X_y=True) 
# print(x.shape)       #(1797, 64)

# lda = LinearDiscriminantAnalysis()    
# x_lda= lda.fit_transform(x, y) 
# print(x_lda.shape)   #(1797, 9)


# lda_EVR = lda.explained_variance_ratio_
# cumsum = np.cumsum(lda_EVR)
# print(cumsum)
# [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662 0.94984789 0.9791736  1.        ]


######[for문 data5가지 LDA_EAR]#######################################################
#1. 데이터
# data_list = [load_iris(return_X_y=True), load_breast_cancer(return_X_y=True),
#              load_wine(return_X_y=True), load_digits(return_X_y=True),
#              fetch_covtype(return_X_y=True)]
data_list = [load_iris, load_breast_cancer,
             load_wine, load_digits, fetch_covtype]
data_name = ['iris', 'cancer', 'wine', 'digits', 'fetch_covtype']

for i, v in enumerate(data_list):
    x,y =v(return_X_y=True)
    print("="*10, data_name[i] ,"="*10)
    # print(x.shape)
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x,y)
    print(x.shape, "==lda==>" ,x_lda.shape)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print('cumsum', cumsum)
print("=============================================")
    
'''
========== iris ==========
(150, 4) ==lda==> (150, 2)
cumsum [0.9912126 1.    ]
========== cancer ==========
(569, 30) ==lda==> (569, 1)
cumsum [1.]
========== wine ==========
(178, 13) ==lda==> (178, 2)
cumsum [0.68747889 1.    ]
========== digits ==========
(1797, 64) ==lda==> (1797, 9)
cumsum [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662 0.94984789 0.9791736  1.  ]
========== fetch_covtype ==========
(581012, 54) ==lda==> (581012, 6)
cumsum [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1. ]
=============================================
'''


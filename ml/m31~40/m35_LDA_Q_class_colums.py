# Q. 컬럼의 개수가 클래스의 개수보다 작을때 디폴트로 돌아가느냐?
# A. 열의 개수가 클래스의 개수보다 작을때에는 열의 개수가 디폴트로 적용됨!!

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
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape) #(50000, 32, 32, 3)

x_train = x_train.reshape(-1, 32*32*3) #(-1, 3072)

pca = PCA(n_components=98)
x_train = pca.fit_transform(x_train)
print(x_train.shape)

lda = LinearDiscriminantAnalysis()    
x_lda= lda.fit_transform(x_train, y_train) 
print(x_lda.shape)   #(50000, 98)  

##디폴트 : 클래스 개수-1 or n_feature 최솟값이 디폴트로 나옴 
#A. 따라서, 열의 개수가 클래스의 개수보다 작을때에는 열의 개수가 디폴트로 적용됨!!


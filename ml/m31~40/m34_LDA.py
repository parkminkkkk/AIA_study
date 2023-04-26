# Linear Discriminant Analsis
# <LDA> 
# -각 데이터의 클래스 별로 맵핑함(매치시킴)
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

# # 1. 데이터
# x, y = load_iris(return_X_y=True) #(150,4)
# # x, y = load_digits(return_X_y=True) #(1797, 64) => (1797, 9) : #디폴트 : 클래스 개수-1 (원래 클래스 개수 10개)

# # pca = PCA(n_components=3)
# # x = pca.fit_transform(x)
# print(x.shape) #(150,3)

# # lda = LinearDiscriminantAnalysis()   
# lda = LinearDiscriminantAnalysis(n_components=3)  #디폴트 : 클래스 개수-1 or n_feature
# #n_componets는 클래스 개수 빼기 하나 이하로 가능하다.      
# x= lda.fit_transform(x, y) 
# print(x.shape)  #(150, 2) => 총 클래스3개 중에 1개, 2개의 형태로 몰려있으므로 그 클래스 사이로 선을 그었음/ 디폴트에 따라 컬럼모양을 잡으므로 2개 나옴



#############iris 클래스 분포 그래프######################
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#iris 데이터 셋 로드
iris = load_iris()

#데이터셋에서 꽃잎의 길이와 폭 정보 추출
x = iris.data[:, 2:] #petal length, petal width
y = iris.target

#scatter plot 그리기 
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('iris')
plt.show()
#######################################################
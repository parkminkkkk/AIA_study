#Q. 회귀데이터도 LDA가 될까?
#A. 회귀문제는 lda해주면 안됨

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing

# 1. 데이터
# x, y = load_diabetes(return_X_y=True) 
x, y = fetch_california_housing(return_X_y=True) 
# y = np.round(y)
print(y)
# print(len(np.unique(y)))  / #214
print(np.unique(y, return_counts=True))
print(x.shape) #(442, 10) / (20640, 8)


lda = LinearDiscriminantAnalysis()    
x_lda= lda.fit_transform(x, y) 
print(x_lda.shape)   


#############################################################
#load_diabetes
#(442, 10)
#회귀데이터여도 y값이 정수값이므로 클래스의 개수로 판단했음 (잘못 인식함)

#fetch_california_housing
#ValueError: Unknown label type: (array([4.526, 3.585, 3.521, ..., 0.923, 0.847, 0.894]),)
#=> y값에 round처리 :  (20640, 5) / lda적용 되긴 함 
#즉, round처리를 해주면서 정수로 만들어줘서 클래스로 있다고 판단을 함(잘못인식) -> 데이터 조작이 될 수 있음

#따라서, 회귀문제는 lda해주면 안됨
#(정수형이라서 LDA에서 y의 클래스로 잘못 인식해서 돌아 간것이므로 회귀데이터는 원칙적으로는 에러임
# 돌리고 싶으면 돌려도는 되나, 데이터 조작가능성이 있고 성능 보장하지 못함)
#############################################################

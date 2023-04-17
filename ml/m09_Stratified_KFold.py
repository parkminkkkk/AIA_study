# m07파일 복붙 
# 문제점 1. train, test/ test로 predict한 것이므로 과적합만큼 결과의acc가 안나올 수 있음 
# 문제점 2. stratify / y값이 편향되는 문제 발생할 수 있음(y의 class의 비율만큼) 
# => StratifiedKFold #y의 클래스만큼 n빵 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2) #stratify=y
 
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337) #stratify=y(동일)

#2. 모델구성
model = SVC()

#3.4 컴파일, 훈련, 평가, 예측 
score = cross_val_score(model, x_train, y_train, cv=kfold)    #model.compile, model.fit, model.evaluate 모두 포함되어있음 
print('cross_val_score(ACC):', score, 
      '\nc_val_score(mean_ACC)', round(np.mean(score), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('c_val_predict acc :', accuracy_score(y_test, y_predict))

# print("=============================================")
# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2]), array([44, 39, 37], dtype=int64))
# print(np.unique(y_test, return_counts=True))  #(array([0, 1, 2]), array([ 6, 11, 13], dtype=int64)) 
# #만약, 0,1,2의 비율이 크게 차이가 난다면? 모델이 한쪽으로 편향될 수 있음 -> 문제 발생 


#비포
# cross_val_score(ACC): [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333] 
# c_val_score(mean_ACC) 0.9583
# c_val_predict acc : 0.8333333333333334   
###test_data로 예측한 값이므로 과적합한 만큼 결과가 안 좋게 나올 수도 있음###

#애프터 StratifiedKFold
# cross_val_score(ACC): [0.875  0.91666667  1.     1.      1.     ] 
# c_val_score(mean_ACC) 0.9583
# c_val_predict acc : 0.9666666666666667   
###y클래스 비율 맞춰주니까 predict_acc 올라감###

#즉, ACC의 문제점 : 편향된 데이터에서는 acc으로 판단하기 어려움 -> 따라서, f1_score로 사용함 

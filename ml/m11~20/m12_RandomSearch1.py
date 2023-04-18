#m10_2,3파일 복붙
#RandomizedSearchCV : gridsearch에서 랜덤하게 뽑아서 사용
#GridSearch : 파라미터 전체를 다 조사해보겠다. (hyperparameter + cross_validation)
#파라미터 : model, model.fit 두가지에 파라미터 사용함 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV    #(hyperparameter + cross_validation)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
import time

#1. 데이터 
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2, stratify=y
)

n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

#gridsearch : 딕셔너리 형태로 입력(각각 입력되기 위해서)
parameters = [
    {"C":[1,10,100,1000], "kernel":['linear'], 'degree':[3,4,5]},   #12번
    {"C":[1,10,100], "kernel":['rbf','linear'], "gamma":[0.001, 0.0001]}, #12번
    {"C":[1,10,100,1000], "kernel":['sigmoid'], 'gamma':[0.01,0.001,0.0001], 'degree':[3,4]} #24번
    ] #총 48번

#2. 모델
 #for문과 동일함(다중for문)
# model = GridSearchCV(SVC(), parameters, #총48 x cv=5 : 240번 돌려라 
model = RandomizedSearchCV(SVC(), parameters,  
                    #cv=kfold,  #kfold쓰는 것보다 cv=5하는게 성능 더 좋을 수 있음.(##stratifiedkfold가 디폴트##)
                     cv=4,      #girdsearch에 cv가 포함되어있음 /즉, 분류모델일때, default가 stratifiedKFold / 회귀일땐, default=kfold
                     verbose=1,  
                     refit=True, #(디폴트)true인 경우 최적의 하이퍼 파라미터를 찾은 후 입력된 개체를 해당 하이퍼 파라미터로 재학습/ false : 최종 파라미터로 출력
                     n_jobs=-1, #전체 cpu다 돌릴거야 
                     n_iter=5,  #디폴트는 10, 디폴트일 경우 [10번*cv]만큼 훈련
                     )  

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수:", model.best_estimator_) 
print("최적의 파라미터:", model.best_params_)
print("best_score:", model.best_score_)
#여기까지는 train으로 확인 - train의 best_score
#test로 확인 - test의 score *더 중요*
print("model.score:", model.score(x_test, y_test))
print("걸린시간 :", round(end_time-start_time,2), "초")

y_predict = model.predict(x_test)
print("accuracy_score:", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)            
print("최적 튠 ACC:", accuracy_score(y_test, y_pred_best))

################################################################
import pandas as pd
# print(model.cv_results_) : 한 눈에 보기 어려움 
# pd : 컬럼 하나(1차원)-Seris,벡터 / list형태 - DataFrame
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)) #값순으로 정렬 : sort_values,ascending=True(오름차순:디폴트)/ false:(내림차순)
#[48 rows x 17 columns] : 48개 훈련에 대해서 17가지 결과값으로 보여줌
print(pd.DataFrame(model.cv_results_).columns) #컬럼명 확인
'''
Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
       'split0_test_score', 'split1_test_score', 'split2_test_score',          #split(0~4):cv의 5번째 점수
       'split3_test_score', 'split4_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score'],
      dtype='object')
'''

path = './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path+'m12_RandomSearch3.csv')
################################################################

#Fitting 4 folds for each of 10 candidates, totalling 40 fits
#cv 1개당 10번씩만 선택해서 돌리겠다. (48개 훈련 중에서 10개만 선택해서 훈련하겠다.)

#n_iter=5일때, (디폴트= n_iter=10)
#Fitting 4 folds for each of 5 candidates, totalling 20 fits
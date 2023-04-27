##함수의 인자자체를 여러개 받을 수 있음(개수에 제한 없음) ##
# *args : * 변수, 값 
# **params : ** 딕셔너리(키:밸류)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier

#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits =5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = {'n_estimators' : 100000,
              'learning_rate' : 0.1,
              'max_depth': 3,
              'gamma': 1,
              'min_child_weight': 1,
              'subsample': 0.7,
              'colsample_bytree': 0.2,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.1,
              'random_state' : 337,
            #   'eval_metric' : 'error'
              }

#2. 모델 
# model = XGBClassifier(**parameters)
model = XGBClassifier()


#3. 훈련
# model.set_params(early_stopping_rounds =10)                                    
model.set_params(early_stopping_rounds =100, eval_metric = 'error', **parameters)    #eval_metric = 'error' set_params에서 따로 명시 or **parameters안에 명시해서 사용가능

#eval_metric = 'logloss'(default), 'error', 'auc' (이진분류)/ 'merror', 'mlogloss':  (multi-)다중분류에서 사용가능(이진분류에서 사용시 에러뜸)
#eval_metric = 'rmse', 'mae', 'rmsle' (회귀평가지표)/ 그러나, 분류는 회귀로 평가가능함(회귀는 분류로 평가xx)

# hist = 
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],     #es사용시, validation 필수 사용
        #   early_stopping_rounds =10,                           #es사용해주면 n_estimators값 키울 수 있음
          verbose =1 
          
          )   

#4. 평가, 예측

results = model.score(x_test, y_test)
print("최종 점수 :", results)
# print(hist)
#modle.fit에 hist하면 파라미터에 대한 정보만 나옴/ ml에서는 다른 곳에 hist있다는 의미 

print("==============================================")
hist = model.evals_result()
# print(hist)

#[그래프]
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic' #한글깨짐 해결 #다른 폰트 필요시 윈도우 폰트파일에 추가해줘야함
plt.figure(figsize=(9,6))
plt.plot(hist['validation_0']['error'], marker='.', c='red') 
plt.plot(hist['validation_1']['error'], marker='.', c='blue')
plt.title('cancer')
plt.xlabel('epochs')
plt.ylabel('val0_error, val1_error')
plt.legend(('val0_error', 'val1_error')) # 범례 표시 
plt.grid()    #격자표시 
plt.show()

# 최종 점수 : 0.9736842105263158
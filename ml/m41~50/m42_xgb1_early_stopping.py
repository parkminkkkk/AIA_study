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

parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.1,
              'max_depth': 6,
              'gamma': 0,
              'min_child_weight': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state' : 337,
            #   'eval_metric' : 'rmse'
              }

#2. 모델 
# model = XGBClassifier(**parameters)
model = XGBClassifier()


#3. 훈련
# model.set_params(early_stopping_rounds =10)
model.set_params(early_stopping_rounds =10, **parameters)    #eval_metric = 'rmse' set_params에서 따로 명시 or **parameters안에 명시해서 사용가능

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],     #es사용시, validation 필수 사용
        #   early_stopping_rounds =10,                           #es사용해주면 n_estimators값 키울 수 있음
          verbose =1 
          )   

#4. 평가, 예측

results = model.score(x_test, y_test)
print("최종 점수 :", results)

'''
[0]     validation_0-logloss:0.50821
[1]     validation_0-logloss:0.41616
[2]     validation_0-logloss:0.34737
[3]     validation_0-logloss:0.29869
[4]     validation_0-logloss:0.26980
[5]     validation_0-logloss:0.24646
.
.
.
[23]    validation_0-logloss:0.22224
[24]    validation_0-logloss:0.22478
[25]    validation_0-logloss:0.22811
[26]    validation_0-logloss:0.22485
[27]    validation_0-logloss:0.22193
최종 점수 : 0.9385964912280702
'''


# error1.
# AssertionError: Must have at least 1 validation dataset for early stopping. 
# =>>> eval_set : es사용시, validation 필수 사용

# error2.
# xgboost.core.XGBoostError: Invalid Parameter format for colsample_bylevel expect float but value='[1]'
# =>>> 'colsample_bylevel': [1] -> 'colsample_bylevel': 1. (리스트형태를 수치형으로 변환해줘야함 )

# UserWarning1.
#UserWarning: `early_stopping_rounds` in `fit` method is deprecated for better compatibility with scikit-learn, use `early_stopping_rounds` in constructor or`set_params` instead.
'''
#model.fit안에 넣지 말고, model.set_params를 따로 빼서 사용해주기
model.set_params(early_stopping_rounds =10)
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],  #es사용시, validation 필수 사용
        #   early_stopping_rounds =10,      #es사용해주면 n_estimators값 키울 수 있음
          verbose = 0
          ) 
'''
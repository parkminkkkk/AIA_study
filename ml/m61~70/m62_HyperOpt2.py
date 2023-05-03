# HyperOpt : 최솟값 찾기 [함수(최솟값 뽑는 함수정의), 파라미터의 범위 준비] 
# 회귀 평가지표 : mse, mae(최솟값) or r2(-최댓값)

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')
# *UserWarning: 'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. 
# Pass 'early_stopping()' callback via 'callbacks' argument instead.

#1. 데이터 
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#hyperopt----------------------------------------------------------#
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK

search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.01, 1),          #uniform(실수형태) : 정규분포형태에 따라서 제공하겠다 (->중심부(0.5부분)를 더 많이 제공하게 됨) =>> q값(범위값) 줄 필요 xx
    'max_depth' : hp.quniform('max_depth',3, 16, 1.0),               #quniform(정수형태) /// 1단위이지만, 사실은 1.0 => int해줘야함/ 그래도 1단위이므로 round는 안해도됨
    'num_leaves' : hp.quniform('num_leaves',24, 64, 1),          
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 2), 
    'min_child_weight' : hp.quniform('min_child_weight',1, 50, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),         
    'colsample_bytree' : hp.uniform('colsample_bytree',0.5, 1),
    'max_bin' : hp.quniform('max_bin',9, 500, 2),            
    'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10),      
    'reg_alpha' : hp.uniform('reg_alpha',0.01, 50)
}
# hp.uniform(label, low, high) : 최소부터 최대까지 정규분포 간격
# hp.quniform(label, low, high, q) : 최소부터 최대까지 q간격
# hp.randint(label, upper) : 0부터 최댓값upper(지정)까지 random한 정수값
# hp.loguniform(label, low, high) : exp(uniform(low, high))값 반환  /이것 또한 정규분포/  log변환 한것을 다시 지수로 변환한다(exp)
# *x : 독립변수 (x의 값이 너무 크거나 치우쳐져있는 경우에도 log변환) / y : 종속변수( 주로 y값 log변환) 


#모델 정의 
def lgbm_hamsu(search_space):
    params = {
        'n_estimators' : 1000,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'],1),0),
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']),10),
        'reg_lambda' : max(search_space['reg_lambda'],0),
        'reg_alpha' : search_space['reg_alpha'],
        
    }

    model = LGBMRegressor(**params)
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    y_predict = model.predict(x_test)
    return_value = mean_squared_error(y_test, y_predict)

    return return_value

trial_val = Trials()   #hist보기위해


best = fmin(
    fn= lgbm_hamsu,                            #함수
    space= search_space,                        #파라미터
    algo=tpe.suggest,                           #알고리즘 정의(디폴트) // 베이지안 최적화와의 차이점..
    max_evals=50,                               #베이지안 최적화의 n_iter와 동일(훈련 10번)
    trials=trial_val,                           #결과값 저장
    rstate = np.random.default_rng(seed=10)    #random_state와 동일
)


print("best:", best)
# best: {'colsample_bytree': 0.5460817592066081, 'learning_rate': 0.26687539574943836, 'max_bin': 64.0, 'max_depth': 11.0, 
# min_child_samples': 92.0, 'min_child_weight': 22.0, 'num_leaves': 33.0, 'reg_alpha': 28.361416061697895, 'reg_lambda': 7.394361180530024, 'subsample': 0.8242331462770404}


results = [aaa['loss'] for aaa in trial_val.results]   #trial_val.results의 값을 aaa반복해라 / aaa의 ['loss']만 반복해라 
df = pd.DataFrame({
        'learning_rate' : trial_val.vals['learning_rate'],
        'max_depth' : trial_val.vals['max_depth'],
        'num_leaves' : trial_val.vals['num_leaves'],
        'min_child_samples' : trial_val.vals['min_child_samples'],
        'min_child_weight' : trial_val.vals['min_child_weight'],
        'subsample' : trial_val.vals['subsample'],
        'colsample_bytree' : trial_val.vals['colsample_bytree'],
        'max_bin' : trial_val.vals['max_bin'],
        'reg_lambda' : trial_val.vals['reg_lambda'],
        'reg_alpha' : trial_val.vals['reg_alpha'],
         'results': results
                   })
print(df)

### results칼럼에 최솟값이 있는 행을 출력 ###
min_row = df.loc[df['results'] == df['results'].min()]
print("최소 행",'\n' , min_row)
#     learning_rate  max_depth  num_leaves  min_child_samples  min_child_weight  subsample  colsample_bytree  max_bin  reg_lambda  reg_alpha     results
# 5       0.266875       11.0        33.0               92.0              22.0   0.824233          0.546082     64.0    7.394361  28.361416  2557.74156

### results칼럼에 최솟값이 있는 행에서 results만 출력 ###
min_results = df.loc[df['results'] == df['results'].min(), 'results']
print(min_results.values)  #[2557.74155988]
#print(min(results))       #2557.741559883484
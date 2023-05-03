import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization


lgbm_params = {
    'max_depth' : (3,16),
    'num_leaves' : (24,64),
    'min_child_samples' : (10, 200), 
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)
}


#1. 데이터 
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, #stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_splits =5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)


#2. 모델 
lgbm = LGBMRegressor(random_state =337)
optimizer = BayesianOptimization(lgbm, lgbm_params, n_jobs=-1)

optimizer.maximize(init_points=2,   #두개의 랜덤값
                   n_iter=20,       #20번 파라미터 검색 횟수(훈련)        => 총 22번 
                   )
print(optimizer.max)



#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 :", model.best_score_)

results = model.score(x_test, y_test)
print("최종 점수 :", results)

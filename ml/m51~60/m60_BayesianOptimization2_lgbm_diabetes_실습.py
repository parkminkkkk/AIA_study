import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score

 #파라미터를 실수로 받아들임-> 정수로 변환 해줘야함
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


def lgbm_model(max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    model = LGBMRegressor(
        random_state=337,
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        min_child_weight=int(min_child_weight),
        subsample=subsample,                    #float
        colsample_bytree=colsample_bytree,
        max_bin=int(max_bin),
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha
    )

    x, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=337, train_size=0.8
    )
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return r2
    


optimizer = BayesianOptimization(lgbm_model, pbounds = lgbm_params)

optimizer.maximize(init_points=10, n_iter=500)
print(optimizer.max)

'''
{'target': 0.49674070364471323, 'params': {'colsample_bytree': 0.5, 'max_bin': 82.49469535910023, 'max_depth': 3.0, 'min_child_samples': 78.76941909768044, 'min_child_weight': 1.0, 'num_leaves': 64.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 1.0}}  
  (r2)
'''

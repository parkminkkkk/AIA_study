# 각 파라미터별로 최상값 찾으면서 돌리기 
# xgb 에서 learning_rate, max_depth 가장 많이 사용하는 파라미터
# 'learning_rate'파라미터에 따라 성능이 크게 차이남 (성능에 영향을 많이 주는 파라미터)
# 너무 작아도 최솟값까지 도달하지 못하고, 너무 커도 최솟값 주변에서 돌고있게 됨 
# 'max_depth' : 트리의 최대 깊이 (깊을수록 복잡한 패턴학습하나 과적합 주의) 

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

parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
              'learning_rate' : [0.3],
              'max_depth': [6],
              'gamma': [0],
              'min_child_weight': [1],
              'subsaample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1],
              'reg_alpha': [0],
              'reg_lambda': [1]
              }

#2. 모델 
xgb = XGBClassifier(random_state =337)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)  

#4. 평가, 예측
print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 :", model.best_score_)

results = model.score(x_test, y_test)
print("최종 점수 :", results)


'''
최상의 매개변수 : {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1, 'subsaample': 1}
최상의 점수 : 0.9604395604395604
최종 점수 : 0.9473684210526315
'''


# xgboost 파라미터 
# 'n_estimators' : [100, 200, 300, 400, 500, 1000]  /디폴트100/ 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] / 디폴트0.3 / 0~1/ eta
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] /디폴트6/ 0~inf/ 정수 
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] / 디폴트 0/ 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] /디폴트 1/ 0~inf
# 'subsaample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] /디폴트 0/ 0~inf/ L1 절대값 가중치 규제/ alpha
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10] /디폴트 1/ 0~inf/ L2 제곱 가중치 규제/ lambda

# n_estimators: XGBoost 모델에서 생성할 트리의 개수.
# learning_rate: 손실 함수의 최소값으로 이동하는 동안 각 반복에서의 단계 크기
# max_depth: 트리의 최대 깊이. 더 깊은 트리는 더 복잡한 패턴을 학습할 수 있지만 과적합 주의 
# gamma: 트리의 복잡도를 조절하기 위한 파라미터. 값이 높을수록 분할 수가 줄어듦
# min_child_weight: 새 노드를 생성하기 위해 자식에 필요한 인스턴스 가중치(hessian)의 최소 합. 과적합 방지에 도움
# subsample: 각 트리에 대해 임의로 샘플링할 인스턴스의 비율. 과적합 방지에 도움
# colsample_bytree: 각 트리에 대해 임의로 샘플링할 기능의 비율. 과적합 방지에 도움
# colsample_bylevel: 각 수준에서 각 분할에 대한 열의 하위 표본 비율.
# colsample_bynode: 각 노드에서 각 분할에 대한 열의 하위 샘플 비율.
# reg_alpha: 가중치에 대한 L1 정규화 용어. 기능 선택에 사용가능.
# reg_lambda: 가중치에 대한 L2 정규화 용어. 과적합 방지에 도움.

#머신러닝모델
#L1규제 : 절대값(라쏘)
#L2규제 : 제곱(릿지)
#layer상에서 양수로 만들겠다. 

#feature_importances 
#트리계열에서만 제공함!! model.feature_importances_
#컬럼의 종류에 따라 훈련결과에 악영향을 주는 불필요한 컬럼 존재함 (노이즈)
#=> 컬럼(열, 특성, feature) 걸러내는 작업 필요함 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline

#1. 데이터 
datasets = [load_diabetes,
            fetch_california_housing] 
dataname = ['diabetes','california']


model = [DecisionTreeRegressor(), RandomForestRegressor()]
modelname = ['DTR', 'RFR']

for j, v1 in enumerate(datasets):
    x, y = v1(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    # print(f'Data: {data_name}')
    for i, v in enumerate(model):
        model = v
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
    print(type(model).__name__, ":", "r2:", r2)
    print(type(model).__name__, ":", "컬럼별 중요도", model.feature_importances_)
    print('-------------------------------------------')


# RFR : r2: 0.40249684276988296
# RFR : 컬럼별 중요도 [0.06890791 0.00884723 0.27451089 0.08078943 0.04824584 0.05678677
#  0.04978026 0.02219364 0.32151868 0.06841935]
# -------------------------------------------


#####################################
#2. 모델구성
#공통점 : 트리 계열(분류) 
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

# #3. 훈련 
# model.fit(x_train, y_train)

# #4. 평가, 예측
# result = model.score(x_test, y_test)
# print("model.score:", result)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score:", acc)

# print(model, ":", model.feature_importances_)

#1.
# model.score: 0.9666666666666667
# accuracy_score: 0.9666666666666667
# DecisionTreeClassifier() : [0.         0.01671193 0.93062443 0.05266364]  
# ==================================================
# iris: 컬럼4개_이에 대한 중요도 // 0.93이 나온 데이터에서 각 컬럼별로 중요도를 미친 정도(=0.93값의 컬럼별 기여도) 
# (따라서, 0.93점수를 신뢰한다는 전제조건)

#2. 
# model.score: 0.9666666666666667
# accuracy_score: 0.9666666666666667
# RandomForestClassifier() : [0.11048247 0.02531084 0.44641015 0.41779654]

#3. 
# model.score: 0.9666666666666667
# accuracy_score: 0.9666666666666667
# GradientBoostingClassifier() : [0.0061731  0.01334182 0.7564738  0.22401128]

#4. 
# model.score: 0.9666666666666667
# accuracy_score: 0.9666666666666667
# XGBClassifier() : [0.01794496 0.01218657 0.8486943  0.12117416]


#model.feature_importances_
#=> 모두 3번째 컬럼의 중요도가 가장 높음. 
#=> 1,2번째 컬럼은 삭제해도 97%의 성능은 기대할 수 있음 (오히려 성능 늘어날 수도 있음.)
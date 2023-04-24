# [실습]
# feature_importance가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성 후 모델 돌려서 결과 도출
# 기존 모델들과 성능 비교 
# 모델 4개 구성

#결과비교 (삭제전-삭제후)
'''
ex) 
1. DT
#기존acc : 
#컬럼 삭제후 acc: 
2. RF
3. GDB
4. XGB
'''
####################################################
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

def runmodel(x_train,x_test,y_train,y_test,model:GradientBoostingClassifier,data_name:str):
    model.fit(x_train, y_train)
    fi_result = model.score(x_test, y_test)
    print(data_name, type(model).__name__, 'fi result : ', fi_result)
    y_fi_pred = model.predict(x_test)
    print(data_name, type(model).__name__, 'fi acc : ', accuracy_score(y_test, y_fi_pred))

data_list = [load_iris, load_breast_cancer, load_wine, load_digits, fetch_california_housing, load_diabetes]
classifier_model_list = [RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, XGBClassifier]
regressor_model_list = [RandomForestRegressor(), DecisionTreeRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if i<4:
        for j in range(len(classifier_model_list)):
            model = classifier_model_list[j]()
            runmodel(x_train,x_test,y_train,y_test,model=model,data_name=data_list[i].__name__)
            fi = model.feature_importances_
            x_fi_train, x_fi_test = x_train, x_test
            index_k = []
            for k in range(len(fi)):
                if fi[k]<np.percentile(fi,25):
                    index_k.append(k)
            x_fi_train, x_fi_test = pd.DataFrame(x_fi_train).drop(index_k, axis=1), pd.DataFrame(x_fi_test).drop(index_k, axis=1)
            runmodel(x_fi_train,x_fi_test,y_train,y_test,model=model,data_name=data_list[i].__name__)
            
    if 4<=i:
        for j in range(len(regressor_model_list)):
            model = regressor_model_list[j]
            runmodel(x_train,x_test,y_train,y_test,model=model,data_name=data_list[i].__name__)
            fi = model.feature_importances_
            x_fi_train, x_fi_test = x_train, x_test
            index_k = []
            for k in range(len(fi)):
                if fi[k]<np.percentile(fi,25):
                    index_k.append(k)
            x_fi_train, x_fi_test = pd.DataFrame(x_fi_train).drop(index_k, axis=1), pd.DataFrame(x_fi_test).drop(index_k, axis=1)
            runmodel(x_fi_train,x_fi_test,y_train,y_test,model=model,data_name=data_list[i].__name__)
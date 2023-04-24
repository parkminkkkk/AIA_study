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
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

num_classes = 3
params = {'objective': 'multi:softmax', 
          'num_class' : num_classes}

#1. 데이터

data_list = [load_iris(), load_breast_cancer(), load_digits(), load_wine()]
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(),
               XGBClassifier(objective='multi:softmax', num_class=num_classes)]
# scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()] 

for i in range(len(data_list)):
    data = data_list[i]
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state= 337, stratify=y
    )

    for j, model in enumerate(model_list): 
            # 모델 훈련
            model.fit(x_train, y_train)

            # feature importance 계산
            feature_importance = model.feature_importances_
            feature_importance_percent = feature_importance / feature_importance.sum()

            # 중요도가 낮은 컬럼 제거
            threshold = np.percentile(feature_importance_percent, 25) # 하위 25%
            feature_idx = np.where(feature_importance_percent >= threshold)[0]
            selected_x_train = x_train[:, feature_idx]
            selected_x_test = x_test[:, feature_idx]

            # 모델 훈련 후 정확도 계산
            model.fit(selected_x_train, y_train)
            train_acc = model.score(selected_x_train, y_train)
            test_acc = model.score(selected_x_test, y_test)

            # 결과 출력
            print(f"Dataset: {data_list[i].DESCR.splitlines()[0]}")
            print(f"Model: {type(model).__name__}")
            print(f"Selected Features: {len(feature_idx)} / {x_train.shape[1]}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print("=" * 50)

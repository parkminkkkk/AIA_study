#[과제/실습]
# 피처를 한개씩 삭제하고 성능비교
# 10개의 데이터셋, 파일 생성 / 피처를 한개씩 삭제하고 성능비교 
# 모델은 RF로만 한다. 


import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score

dataset = load_iris()
print(type(dataset))  #<class 'sklearn.utils.Bunch'> #x,y로 나눠줘야함 
# print(dataset)
dfx = pd.DataFrame(dataset.data)
dfy = pd.DataFrame(dataset.target)
print(dfx) #[150 rows x 4 columns]
print(dfy) #[150 rows x 1 columns]


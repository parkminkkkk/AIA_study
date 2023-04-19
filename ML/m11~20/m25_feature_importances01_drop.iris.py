#[과제/실습]
# 피처를 한개씩 삭제하고 성능비교
# 10개의 데이터셋, 파일 생성 / 피처를 한개씩 삭제하고 성능비교 
# 모델은 RF로만 한다. 


from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

dataset = load_iris() 
print(dataset)
df = pd.DataFrame(dataset)
dfx = df.data
dfy = df.target

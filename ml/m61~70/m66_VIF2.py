#다중공선성
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
df['target'] = datasets.target
# print(df)

y = df['target']
x = df.drop(['target'], axis=1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) #확인용 scaler

#VIF 다중공선성 ######################################################################################################
# # 다중공선성 유의할점 
# 1. 선 스케일링 필수(통상 StandardScaler)  (왜냐면, 각 컬럼별로 값의 차이가 크니까 비슷하게 scaler로 맞춰주고 확인해야함)
# 2. y넣지 않는다 (vif의 경우에는 y도 영향을 미치는 애들이 있기때문에 넣지 xxx)

vif = pd.DataFrame()
vif['variables'] = x.columns
vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] 
print(vif)

'''
#통상 10이상/ 빡빡한 기준 5이상- 다중공선성이 높다고 판단 
    variables       VIF
0      MedInc  2.501295
1    HouseAge  1.241254
2    AveRooms  8.342786 *
3   AveBedrms  6.994995
4  Population  1.138125
5    AveOccup  1.008324
6    Latitude  9.297624 *
7   Longitude  8.962263 *
'''

x = x.drop('Latitude', axis =1)
# x = x.drop(['Latitude', 'Longitude'], axis =1)
print(x)

vif2 = pd.DataFrame()
vif2['variables'] = x.columns
vif2['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x.shape[1])] 
print(vif2)
'''
    variables       VIF
0      MedInc  2.501295
1    HouseAge  1.241254
2    AveRooms  8.342786
3   AveBedrms  6.994995
4  Population  1.138125
5    AveOccup  1.008324
6   Longitude  9.297624
'''
###################################################################################################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, #stratify=y
)

scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

#2. 모델
model = RandomForestRegressor(random_state=337)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과:", results)


# 결과: 0.8020989209821241
# Latitude_drop 결과: 0.7298013684941245
# ['Latitude', 'Longitude']drop : 결과: 0.6756589579712217





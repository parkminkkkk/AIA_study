#variance_inflation_factor : 다중공선성
#상관관계가 너무 높은 컬럼이 있을 경우 =>> 삭제/차원축소 가능(corr과 유사) / VIF의 지표로 판단 
#따라서, feature_infortance, variance_inflation_factor, corrlelation 세개로 컬럼의 상관관계 파악 가능 => 삭제 또는 차원축소에 이용

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor  #통계적 기법에서 사용

data = {'size' : [30, 35, 40, 45, 50, 45],
        'rooms' : [2, 2, 3, 3, 4, 3],
        'window' : [2, 2, 3, 3, 4, 3],
        'year' : [2010, 2015, 2010, 2015, 2010, 2014],
        'price' : [1.5, 1.8, 2.0, 2.2, 2.5, 2.3]}

df = pd.DataFrame(data)

print(df)
'''
   size  rooms  window  year  price
0    30      2       2  2010    1.5
1    35      2       2  2015    1.8
2    40      3       3  2010    2.0
3    45      3       3  2015    2.2
4    50      4       4  2010    2.5
5    45      3       3  2014    2.3
'''

#다중공선성을 확인하기에는 각 컬럼별로 값의 차이가 크므로 scaler진행 
x = df[['size', 'rooms', 'window', 'year']]
y = df['price']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

################컬럼간의 다중공선성 확인 #####################################################################
#=>(10이하일때, 다중공선성이 높지 않다고 판단/ 큰값부터 하나씩 제거해보면서 확인 함)
#x_scaled의 컬럼의 개수만큼 for문 돌린 값이 aaa(variance_inflation_factor)안으로 들어감 => 그렇게 나온 값을 vif[VIF]에 넣어줘
vif = pd.DataFrame()
vif['variables'] = x.columns
vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] 
print(vif)
'''
  variables         VIF
0      size  378.444444
1     rooms         inf   #infinity : 다중공선성 거의 일치 
2    window         inf
3      year   53.333333
'''
################################################################################################################


print("===================rooms 제거 전=====================")
lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 =r2_score(y, y_pred)
print("r2:", r2)
# r2: 0.9938931297709924

print("===================rooms 제거 후=====================")
x_scaled = df[['size', 'window', 'year']]

vif2 = pd.DataFrame()
vif2['variables'] = x_scaled.columns
vif2['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] 
print(vif2)
'''
  variables         VIF
0      size  295.182375
1    window  139.509263
2      year   56.881874
'''
lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 =r2_score(y, y_pred)
print("r2:", r2)
# r2: 0.9938931297709941     #성능에 불필요한 컬럼 삭제 

print("===================size 제거 후=====================")
x_scaled = df[['window', 'year']]

vif3 = pd.DataFrame()
vif3['variables'] = x_scaled.columns
vif3['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])] 
print(vif3)
'''
  variables        VIF
0    window  17.952005
1      year  17.952005
'''
lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 =r2_score(y, y_pred)
print("r2:", r2)
# r2: 0.9811491171843147
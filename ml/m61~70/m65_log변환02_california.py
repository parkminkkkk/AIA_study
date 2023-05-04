#로그변환 : y값 차이가 너무 많이 날때 log로 변환을 해주면서 큰수를 작은수로 줄여주면서 값의 차이범위는 가져감 
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터셋
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
df['target'] = datasets.target
# print(df)

# df.info()
# print(df.describe())

#이상치 확인 그래프 
# df.boxplot()   
# df.plot.box() #둘다 가능// pd.Series형태일때는 df.plot.box()만 가능!! 
# plt.show()

# df['Population'].boxplot() : AttributeError: 'Series' object has no attribute 'boxplot'
# df['Population'].plot.box()   #Series형태일때, 데이터의 컬럼한개만 뽑아서 그래프 볼때 이거 사용
# plt.show()

# df['Population'].hist(bins=50)  #hist 데이터의 분포확인
# plt.show()

# df['target'].hist(bins=50)      #분위수를 50개로 자르는 거임 
# plt.show()


y = df['target']
x = df.drop(['target'], axis=1)

########### x_population 로그변환 ###########
x['Population'] = np.log1p(x['Population'])      #지수변환 : 1을 빼줌  : np.exp1m (1minus) 
#문제점 : log0은 연산할 수 없음 => 따라서, 연산을 할 수 있도록 1을 더해줌 : np.log1p (1plus)
#로그변환시, 분류형은 힘듦
########### y_population 로그변환 ###########
# y = np.log1p(y)                 
#문제점 : y를 train, test이전에 해주면 마지막에 다시 지수변환 해줘야하므로, 데이터 손상 발생 할 수 있음( split이후에 변환)
############################################


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=337
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)



#모델 
model = RandomForestRegressor(random_state=337)

#컴파일, 훈련 
model.fit(x_train, y_train_log)

#평가, 예측 
score = model.score(x_test, y_test_log)
print("score:", score)



#y데이터의 일부를 변환하지 않은 데이터 준비 or log변환 된것을 다시 지수로 변환, submit제출파일 또한 log변환 된 것이므로 지수변환 해서 제출해야함***
# 지수y_test : np.expml(y_test), 지수y_predict: np.expm1(model.predict(x_test)) -> 이 두개를 비교해야 함 
# print("r2:", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))  #데이터 손상 있을 수 있음

#이것 사용!
print("로그->지수r2:", r2_score(y_test, np.expm1(model.predict(x_test))))            #y로그변환을 train,test split 이후에 따로 해줌




# 로그 변환 전 : score: 0.8021183994602941
# x['pop']만 로그 변환 : score: 0.8022669492389668
# y만 로그 변환 : score: 0.8244322268075517
# x,y 모두 로그 변환 후 : score: 0.824475366593445


# 로그 변환 전 : score: 0.8021183994602941
# r2: 0.6367924334892385             #다시 지수변환한 결과값 
# 로그->지수r2: 0.8006340175132711    #
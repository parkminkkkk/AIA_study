#로그변환 : y값 차이가 너무 많이 날때 log로 변환을 해주면서 큰수를 작은수로 줄여주면서 값의 차이범위는 가져감 
from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터셋
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
df['target'] = datasets.target
# print(df)

df.info()
print(df.describe())

# df.boxplot()   
# df.plot.box() 
# plt.show()



df.hist(bins=50)  #hist 데이터의 분포확인
plt.show()

# df['target'].hist(bins=50)      #분위수를 50개로 자르는 거임 
# plt.show()


y = df['target']
x = df.drop(['target'], axis=1)

########### x 로그변환 ###########
x['sex'] = np.log1p(x['sex']) 
x['s4'] = np.log1p(x['s4']) 



x_train_log, x_test_log, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=337
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)



#모델 
model = RandomForestRegressor(random_state=337)

#컴파일, 훈련 
model.fit(x_train_log, y_train_log)

#평가, 예측 
score = model.score(x_test_log, y_test_log)
print("score:", score)



#y데이터의 일부를 변환하지 않은 데이터 준비 or log변환 된것을 다시 지수로 변환, submit제출파일 또한 log변환 된 것이므로 지수변환 해서 제출해야함***
# 지수y_test : np.expml(y_test), 지수y_predict: np.expm1(model.predict(x_test)) -> 이 두개를 비교해야 함 
# print("로그->지수r2:", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))  #데이터 손상 있을 수 있음

#이것 사용!
print("로그->지수r2:", r2_score(y_test, np.expm1(model.predict(x_test_log))))            #y로그변환을 train,test split 이후에 따로 해줌



# score: 0.3460461667125837
# 로그->지수r2: 0.38089927847000704
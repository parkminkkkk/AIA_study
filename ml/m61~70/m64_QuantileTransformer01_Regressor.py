#회귀로 맹그러 - 회귀데이터, scaler6개 for문 !!!

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer  
#QuantileTransformer : 정규분포로 만듦(StandardScaler) + 이후 사분위로 나눈다음 0-1사이로 만들어준다(MinMaxScaler) // minmax와의 차이점은 MinMax는 범위를 생각하지 않아 가운데 쪽에 모여있게 됨
#이상치에 자유로운 scaler : RobustScaler, MinMaxScaler
### https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#quantiletransformer-uniform-output### scaler정리

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

#1. 데이터 
datalist = [load_diabetes(), fetch_california_housing()]
dataname = ['diabetes', 'california']

scalerlist = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), 
              QuantileTransformer(),    #디폴트 n_quantiles=1000 / 분위수 조절  (*diabetes: the total number of samples(353))
              PowerTransformer(),                       #디폴트 method = 'Yeo-Johnson'// 
            #   PowerTransformer(method = 'Yeo-Johnson'), #디폴트 
            #   PowerTransformer(method = 'Box-Cox')      #음수일때 error발생 가능
              ]
# Yeo-Johnson 및 Box-Cox 변환이 지원 / Box-Cox일때, 음수이면 error발생함 따라서, 파라미터 method = 'Yeo-Johnson'/ method = 'Box-Cox' 나눠서 사용
scalername = ['Standard', 'MinMax', 'MaxAbs', 'Robust', 'QuantileT', 'PowerT']


for i in range(len(datalist)):
    data = datalist[i]
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, train_size=0.8, random_state=337, #stratify=y
    )
    print("=====",[i+1], dataname[i],"=====" )

    for j, scale in enumerate(scalerlist):
        scaler = scale
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        #2. 모델 
        model = RandomForestRegressor()

        #3. 훈련 
        model.fit(x_train, y_train)

        #4. 평가, 예측 
        # print("=====",[i+1], dataname[i],"=====" )
        print(scalername[j], "=> 결과:", round(model.score(x_test, y_test), 4))



'''
===== [1] diabetes =====
Standard => 결과: 0.403
MinMax => 결과: 0.3948
MaxAbs => 결과: 0.4217
Robust => 결과: 0.4202
QuantileT => 결과: 0.4154
PowerT => 결과: 0.4129
===== [2] california =====
Standard => 결과: 0.7998
MinMax => 결과: 0.8003
MaxAbs => 결과: 0.7999
Robust => 결과: 0.7999
QuantileT => 결과: 0.7981
PowerT => 결과: 0.7978
'''
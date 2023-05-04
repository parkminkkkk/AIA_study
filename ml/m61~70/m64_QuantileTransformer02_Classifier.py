#분류로 맹그러 - 분류데이터, scaler6개 for문 !!!

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer  

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

#1. 데이터 
datalist = [load_iris(), load_breast_cancer(),load_wine(), load_digits(), fetch_covtype() ]
dataname = ['iris', 'cancer', 'wine', 'digits', 'fetch']

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
        x, y, shuffle=True, train_size=0.8, random_state=337, stratify=y
    )
    print("=====",[i+1], dataname[i],"=====" )

    for j, scale in enumerate(scalerlist):
        scaler = scale
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        #2. 모델 
        model = RandomForestClassifier()

        #3. 훈련 
        model.fit(x_train, y_train)

        #4. 평가, 예측 
        # print("=====",[i+1], dataname[i],"=====" )
        print(scalername[j], "=> 결과:", round(model.score(x_test, y_test), 4))


'''
===== [1] iris =====
Standard => 결과: 0.9333
MinMax => 결과: 0.9667
MaxAbs => 결과: 0.9667
Robust => 결과: 0.9667
QuantileT => 결과: 0.9333
PowerT => 결과: 0.9667
===== [2] cancer =====
Standard => 결과: 0.9474
MinMax => 결과: 0.9474
MaxAbs => 결과: 0.9474
Robust => 결과: 0.9386
QuantileT => 결과: 0.9474
PowerT => 결과: 0.9561
===== [3] wine =====
Standard => 결과: 1.0
MinMax => 결과: 1.0
MaxAbs => 결과: 1.0
Robust => 결과: 1.0
QuantileT => 결과: 1.0
PowerT => 결과: 1.0
===== [4] digits =====
Standard => 결과: 0.9833
MinMax => 결과: 0.9722
MaxAbs => 결과: 0.9806
Robust => 결과: 0.9778
QuantileT => 결과: 0.9806
PowerT => 결과: 0.9806
===== [5] fetch =====
Standard => 결과: 0.9567
MinMax => 결과: 0.9556
MaxAbs => 결과: 0.9561
Robust => 결과: 0.9562
QuantileT => 결과: 0.9564
PowerT => 결과: 0.956
'''




import numpy as np
from sklearn.covariance import EllipticEnvelope
import pandas as pd
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

#1. 데이터
#1. 데이터
path = 'd:/study/_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

print(type(x))
x = x.values

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75], axis=0)
    print('1사분위 : ', quartile_1) 
    print('q2 : ', q2) 
    print('3사분위 : ', quartile_3) 
    iqr = quartile_3 - quartile_1 
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5) 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
outliers_loc = outliers(x)
print('이상치의 위치 : ', list((outliers_loc)))
x[outliers_loc] = np.nan
print(x[outliers_loc])

import matplotlib.pyplot as plt
plt.boxplot(x)
plt.show()

imputer = IterativeImputer(estimator=XGBRegressor())
x = imputer.fit_transform(x)

xgb = XGBRegressor()
xgb.fit(x, y)
results = xgb.score(x,y)
y_submit = xgb.predict(test_csv)
print(results)


submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_bike.csv') # 파일생성


'''
1. [1737_es] 1.35509점
Epoch 00198: early stopping/ loss :  23314.111328125/ r2스코어 : 0.29477181122628304/ RMSE :  152.68958433137038
-patience=20, random_state=34553, Dense(16'relu',8'relu', 4'relu',1), 'mse'

2. [] 
Epoch 01213: early stopping/ loss :  21073.4609375/ r2스코어 : 0.31799011805599553/RMSE :  145.16701284560716
-patience=300, random_state=650874, Dense(8'relu',4'relu', 8'relu',4'relu', 1), 'mse', batch_size=10

*StandScaler, MaxAbsScaler : 4.1점

'''
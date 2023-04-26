#이상치 확인=> 대체 후 결과값 확인 
#이상치->결측치 처리(Nan) -> 결측치 처리(대체)
#dacon_ddarung 
#dacon_diabetes
#kaggle_bike
#kaggle_wine
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.covariance import EllipticEnvelope  #reshape필요 
from xgboost import XGBRegressor

#1. 데이터
path = './_data/dacon_ddarung/'
path_save = './_save/dacon_ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) #(1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) #(715, 9) count제외

# print(train_csv.columns)
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object')
'''
# print(train_csv.info())
# print(train_csv.describe())
# print(type(train_csv))
# print(train_csv.isnull().sum())

# train_csv = train_csv.to_numpy()

# ###결측치제거### 
# train_csv = train_csv.dropna()   #결측치 삭제함수 .dropna()
# print(train_csv.isnull().sum())
# # print(train_csv.info())
# print(train_csv.shape)  #(1328, 10)

###결측치처리###
imputer = IterativeImputer(estimator=XGBRegressor())
train_csv = imputer.fit_transform(train_csv)
test_csv = imputer.fit_transform(test_csv)

train_csv = pd.DataFrame(train_csv)
test_csv = pd.DataFrame(test_csv)
train_csv.columns = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation','hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
test_csv.columns = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation','hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
print(train_csv)  
print(test_csv)

###이상치 처리### 
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']
x=x.to_numpy()

# assuming x is the original dataset
q1 = np.percentile(x, 25)
q3 = np.percentile(x, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5*iqr
upper_bound = q3 + 1.5*iqr

# find the indices of the outliers
outlier_idx = np.where((x < lower_bound) | (x > upper_bound))
print(outlier_idx)

# remove the outliers from the original dataset
x_no_outliers = np.delete(x, outlier_idx)
print(x_no_outliers)

'''
#이상치 찾는 함수(df)
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
print('이상치의 위치 : ', list(outliers_loc))

outliers_loc = np.nan
print(x.info())
'''
'''
#이상치 찾는 함수(df)
# def outliers(aaa):
#     b = []
#     for i in range(aaa.shape[1]):
#         q1, q2, q3 = np.percentile(aaa[:, i], [25, 50, 75], axis=0)
#         print(f'q1 :  {q1}\n q2 : {q2}\n q3 : {q3}')
#         iqr = q3 - q1
#         print('iqr : ', iqr)
#         lower_bound, upper_bound = q1 - (iqr * 1.5), q3 + (iqr * 1.5)
#         b.append(np.where((aaa[:,i]>upper_bound)|(aaa[:,i]<lower_bound)))
#     return b
# outliers_loc = outliers(x)
# print('location of outliers : ', outliers_loc)
# import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()
----------------------------------------------------------------------
#이상치 확인 후 처리
# outlier_idx = outliers(x)[0][0]
# print('이상치확인용데이터x:', outlier_idx)
#1)이상치 삭제
# x = np.delete(x, outlier_idx, axis=0)
# print('이상치 삭제데이터x:',x)
# print('이상치 삭제데이터x:',x.shape)
# #2)중앙값으로 대체
# outlier_idx = outliers(x)[0][0]
# median_val = np.median(x[:,0])
# x[outlier_idx,0] = median_val
# print(x[outlier_idx,0])

# #3)Nan값 대체
# outlier_idx = outliers(x)[0][0]
# x[outlier_idx,0] = np.nan
# print(x[outlier_idx,0])
'''

# apply 함수를 통하여 각 값의 이상치 여부를 찾고 새로운 열에 결과 저장
# def is_kor_outlier(df):
# df['국어_이상치여부'] = df.apply(is_kor_outlier, axis = 1)
# x['outlier'] = x.apply(outliers, axis = 1)

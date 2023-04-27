#Dancon_wine 
#결측치/ 원핫인코딩, 데이터분리, 스케일링/ 함수형,dropout
#다중분류 - softmax, categorical

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier

#1. 데이터 
path = 'd:/study/_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]
print(train_csv.shape) #(5497,13)
 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)   #[1 0 1 ... 1 1 1]
print(type(aaa))  #<class 'numpy.ndarray'>
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white'])) #[0 1]


#1-1 결측치 제거 
# print(train_csv.isnull().sum())
# print(train_csv.info())
# train_csv = train_csv.dropna() #결측치없음 

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']
print(type(y))
print(y)
print("y_shape:", y.shape)           #(5497,)
print('y의 라벨값 :', np.unique(y))  #[3 4 5 6 7 8 9]
# test_csv = test_csv.drop(['type'], axis=1)


#1-2 one-hot-encoding
print('y의 라벨값 :', np.unique(y))  #[3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True)) # array([  26,  186, 1788, 2416,  924,  152, 5]

import pandas as pd
y=pd.get_dummies(y)
y = np.array(y)
print(y.shape)                       #(5497, 7)


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

print(outliers_loc)
print('이상치의 위치 : ', list(outliers_loc))

# outliers_loc = 9999
x[outliers_loc[0].tolist()] = np.nan
import matplotlib.pyplot as plt
plt.boxplot(x)
plt.show()


#1-3 데이터분리 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=640874)

#1-4 스케일링 
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
# print(np.min(x_test), np.max(x_test)) 


#2. 모델구성 

xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4, random_state = 1)


#3. 컴파일, 훈련 

#시간저장
import datetime 
date = datetime.datetime.now()  #현재시간 데이터에 넣어줌
print(date)  #2023-03-14 11:15:21.154501
date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute
#시간을 문자데이터로 바꿈 : 문자로 바꿔야 파일명에 넣을 수 있음 
print(date) #0314_1115



xgb.fit(x_train, y_train) #, mcp])
  
#4. 평가예측 

y_pred = xgb.predict(x_test)

xgb.score(x_train, y_train)

acc = accuracy_score(y_test, y_pred)
print('accuracy_score:', acc)

# Evaluate performance on training set
y_train_pred = xgb.predict(x_train)
print("Training R2 score:", r2_score(y_train, y_train_pred))
print("Training MSE:", mean_squared_error(y_train, y_train_pred))


#submission.csv 만들기 
y_submit = xgb.predict(test_csv)
# print(y_submit)

y_submit = np.argmax(y_submit, axis=1)
# print(y_submit.shape)
y_submit += 3

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = y_submit
# print(submission)
submission.to_csv(path_save + 'submit_xgb_'+ date+ '.csv') # 파일생성


'''
*MM
results: [1.0840587615966797, 0.5518181920051575]
accuracy_score: 0.5518181818181818
*MM
results: [1.034911036491394, 0.5600000023841858]
accuracy_score: 0.56
*MM
results: [0.9965323209762573, 0.5727272629737854]
accuracy_score: 0.5727272727272728
*MM- Dense,dropout 추가
results: [0.9915247559547424, 0.5754545331001282]
accuracy_score: 0.5754545454545454
*MM- Dense,dropout 추가/ acc(Max)

*xgb
accuracy_score: 0.5618181818181818
Training R2 score: 0.9458898371785216
Training MSE: 0.004256148672796387

'''

#판다스 행삭제, 특정값 

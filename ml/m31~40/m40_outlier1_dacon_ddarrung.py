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

###이상치 처리### 
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']
#결측치 선 처리
imputer = IterativeImputer(estimator=XGBRegressor())
x = imputer.fit_transform(x)

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
x[outliers_loc] = np.nan
import matplotlib.pyplot as plt
plt.boxplot(x)
plt.show()


# ###결측치제거### 
# train_csv = train_csv.dropna()   #결측치 삭제함수 .dropna()
# print(train_csv.isnull().sum())
# # print(train_csv.info())
# print(train_csv.shape)  #(1328, 10)
imputer = IterativeImputer(estimator=XGBRegressor())
x = imputer.fit_transform(x)

train_csv =x
train_csv = pd.DataFrame(train_csv)
test_csv = pd.DataFrame(test_csv)
train_csv.columns = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation','hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
test_csv.columns = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation','hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
print(train_csv)  
print(test_csv)

###데이터분리(train_set)###
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640874, test_size=0.2
)
print(x_train.shape, x_test.shape) # (929, 9) (399, 9) * train_size=0.7, random_state=777일 때 /count제외
print(y_train.shape, y_test.shape) # (929,) (399,)     * train_size=0.7, random_state=777일 때 //count제외

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler() 
scaler.fit(x_train) #x_train범위만큼 잡아라
x_train = scaler.transform(x_train) #변환
#x_train의 변환 범위에 맞춰서 하라는 뜻이므로 scaler.fit할 필요x 
x_test = scaler.transform(x_test) #x_train의 범위만큼 잡아서 변환하라 
test_csv = scaler.transform(test_csv) 


#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=9))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min',
              verbose=1, 
              restore_best_weights=True)

hist = model.fit(x_train, y_train,
          epochs=30000, batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )
#print(hist.history['val_loss'])

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

#'mse'->rmse로 변경
import numpy as np
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

#submission.csv 만들기 
y_submit = model.predict(test_csv)
# print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0314_0930_MaxAbScaler.csv') # 파일생성


#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic' #한글깨짐 해결 #다른 폰트 필요시 윈도우 폰트파일에 추가해줘야함
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red') #marker='.' 점점으로 표시->선이됨
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(('로스', '발_로스')) # 범례 표시 
plt.grid()    #격자표시 
plt.show()


'''
##결측치 처리 : IterativeImputer###
점수 : 70.8995636829   *성능향상*
loss :  3098.23681640625
r2스코어 : 0.5376026065728127
RMSE :  55.66180729942712
==================================================
'''

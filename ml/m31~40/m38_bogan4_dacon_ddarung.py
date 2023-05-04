# 결측치처리 활용해서 성능 올려보기 !! -> 최종 성능 비교 

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
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

# ###결측치제거### 
# train_csv = train_csv.dropna()   #결측치 삭제함수 .dropna()
# print(train_csv.isnull().sum())
# # print(train_csv.info())
# print(train_csv.shape)  #(1328, 10)

###결측치처리###
from sklearn.impute import IterativeImputer
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


###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y)

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
1. [1754_es] 73.68점(갱신*)
Epoch 00273: early stopping/ loss :  2475.7314453125/ r2스코어 : 0.5751303841838872/ RMSE :  49.75672111342364
-patience=15, random_state=34553, Dense(16'relu',8'relu', 4'relu',1), 'mse'
2. [1930_es] 67.75점(갱신*)	
Epoch 03301: early stopping/ loss :  2060.399658203125/ r2스코어 : 0.7207531870123005/ RMSE :  45.3916234398938
-patience=300, random_state=640874, Dense(16'relu',8'relu', 4'relu',1), 'mse', batch_Size=10

4. MinMaxScaler() : test_csv파일 scaler!! : 70.54점
Epoch 00566: early stopping
loss :  2336.74853515625
r2스코어 : 0.683299443000897
RMSE :  48.33992993639059

5. *RobustScaler
Epoch3000
loss :  1989.625732421875
r2스코어 : 0.7303451862110664
RMSE :  44.605220624545105
'''
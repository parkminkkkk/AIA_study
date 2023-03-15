#kaggle 바이크 실습 - 결과까지 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd 

#1. 데이터 
path = './_data/kaggle_bike/' 
path_save = './_save/kaggle_bike/' 

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
print(train_csv.shape) #(10886, 11)
#season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_csv)
print(test_csv.shape) #(6493, 8) 
# season  holiday  workingday  weather   temp   atemp  humidity  windspeed

#casual  registered count 3개 차이남 / count는 y값
#현재는 casual(회원x), registered(회원o) 삭제하는게 나음 

#확인절차
#print(train_csv.columns)
'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
'''
#print(test_csv.columns)

#print(train_csv.info())
'''
 0   season      10886 non-null  int64
 1   holiday     10886 non-null  int64
 2   workingday  10886 non-null  int64
 3   weather     10886 non-null  int64
 4   temp        10886 non-null  float64
 5   atemp       10886 non-null  float64
 6   humidity    10886 non-null  int64
 7   windspeed   10886 non-null  float64
 8   casual      10886 non-null  int64
 9   registered  10886 non-null  int64
 10  count       10886 non-null  int64
'''
#print(train_csv.describe())
#print(type(train_csv))
'''
<class 'pandas.core.frame.DataFrame'>
'''
#결측치제거 
print(train_csv.isnull().sum()) #isnull이 True인것의 합계 : 각 컬럼별로 결측치 몇개인지 알수 있음
#결측치 없음 

#train_csv데이터에서 x,y데이터 분리 
x = train_csv.drop(['casual','registered','count'], axis=1)  # drop([],[],[])하면 모델링 안됨..
print(x)
y = train_csv['count']  
print(y)

x_train, x_test, y_train, y_test = train_test_split(
      x, y, shuffle=True, train_size=0.8, random_state=124117
      )

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation= 'linear')) #디폴트값
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
#활성화 함수(한정화함수,activation) : 다음 레이어로 전달하는 값을 한정시킨다 (ex,0-1로 한정하고 싶다)
#'relu함수' : 0이상의 값은 양수, 0이하의 값은 그대로(0) => 따라서, 항상 이후의 값은 양수가 됨/ 히든레이어부분에 대체로 넣음
#'linear'   : 디폴트값 

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=optimizer)
model.fit(x_train,y_train, epochs=1000, batch_size=100,
          verbose=3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
print(submission)

path_save = './_save/kaggle_bike/' 
submission.to_csv(path_save + 'submit_0307_1507.csv') #파일생성

'''
1. RMSE :  151.3396569299613/r2 스코어 : 0.2587575598223131/ loss :  22903.689453125
-결과값[count] 음수나옴/ 값이 랜덤해서 부정확함 => 활성화함수 추가 activation='relu'

2. RMSE :  151.22339123890112/r2 스코어 : 0.2598960315524781/ loss :  22868.51171875
-1.37361점/ train_size=0.8, random_state=650874, Dense(32,16,8 'relu',4 'relu',1), activation='relu' mse, epoch=100, batch_size=50
3.[1235] RMSE :  202.00192979685312/ r2 스코어 : -0.3205833692601274/ loss :  40804.77734375
-1.41667점/ train_size=0.8, random_state=650874, Dense(32,16 'relu',8 'relu',4, 2 'relu',1), activation='relu' mse, epoch=1000, batch_size=100
4.[1301] RMSE :  148.4545206288392/ r2 스코어 : 0.3478544378188214/  loss :  22038.75
-1.28791점/ train_size=0.8, random_state=124117, Dense(32,16 'relu',8 'relu',4, 2 'relu',1), activation='relu' mse, epoch=1000, batch_size=100
5.[1309] RMSE :  149.8061644965068/ r2 스코어 : 0.319137221800734/ loss :  22441.890625
-1.30352점/ train_size=0.8, random_state=34553, Dense(32, 16'relu', 8, 4 'relu', 1), activation='relu' mse, epoch=500, batch_size=10
6.[1435] RMSE :  151.04982950667105/ r2 스코어 : 0.30778548515616444/ loss :  22816.052734375
-1.30352점/ train_size=0.8, random_state=34553, Dense(64, 32'relu', 16, 8 'relu', 1), activation='relu' mse, epoch=500, batch_size=32

*optimizer = Adam(lr=0.0001)
7.[1443] RMSE :  145.13988716356255/ r2 스코어 : 0.3330689940989081/ loss :  21065.587890625
-1.30352점/ train_size=0.8, random_state=2579, Dense(64'relu', 32'relu', 16'relu', 8, 4 'relu', 1), activation='relu' mse, epoch=500, batch_size=32

'''

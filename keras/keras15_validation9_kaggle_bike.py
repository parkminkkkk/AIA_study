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
model.add(Dense(64, input_dim=8))
model.add(Dense(12, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation= 'linear')) #디폴트값
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=100,verbose=1,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

#'mse'->rmse로 변경
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

#submission.csv 만들기 
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
print(submission)

path_save = './_save/kaggle_bike/' 
submission.to_csv(path_save + 'submit_0307_1900_val.csv') #파일생성

'''
[1900_val]
loss :  66657.3984375
r2 스코어 : -1.0163183167096426
RMSE :  258.1809109996835
val=0.2
train_size=0.8, random_state=124117, Dense(64, 12 'relu',32 'relu',16, 4 'relu',1), activation='relu' mse, epoch=100, batch_size=100

[1900] 
loss :  23589.953125
r2 스코어 : 0.2864278237198271
RMSE :  153.5902102898565
train_size=0.8, random_state=124117, Dense(64, 12 'relu',32 'relu',16, 4 'relu',1), activation='relu' mse, epoch=100, batch_size=100
'''
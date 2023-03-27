import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #(10886, 11)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #(6493, 8)  /casual(비회원)  registered(회원) count 3개 차이남

# print(train_csv.columns)
# print(train_csv.info())
# print(train_csv.describe())
# print(type(train_csv))
# print(train_csv.isnull().sum()) 

###결측치제거### 
print(train_csv.isnull().sum()) #isnull이 True인것의 합계 : 각 컬럼별로 결측치 몇개인지 알수 있음
#결측치 없음
# print(train_csv.info())
#print(train_csv.shape)  #(10886, 11)

###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=34553, test_size=0.2
)
print(x_train.shape, x_test.shape) #(8708, 8) (2178, 8) / casual(비회원)  registered(회원) count 3개 drop
print(y_train.shape, y_test.shape) #(8708,) (2178,)

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
#test_csv파일또한 scaler해줘야함! 아니면 제출했을때 점수이상하게 나옴 (train파일을  scale한만큼, test파일도 scale해줘야함)
#train_csv파일에서 x_train,x_test값 가져온것이기 때문에, test_csv파일 scale해줘야함 

#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=8))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
              verbose=1, 
              restore_best_weights=True)

hist = model.fit(x_train, y_train,
          epochs=5000, batch_size=100,
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
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

#submission.csv 만들기 
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0313_1830_RbScaler.csv') # 파일생성


'''
1. [1737_es] 1.35509점
Epoch 00198: early stopping/ loss :  23314.111328125/ r2스코어 : 0.29477181122628304/ RMSE :  152.68958433137038
-patience=20, random_state=34553, Dense(16'relu',8'relu', 4'relu',1), 'mse'

2. [1030_mmscaler] *MinMaxScaler(): test_csv파일 scale안함   : 4.13점 (XXX)
Epoch 02015: early stopping/ loss :  23302.654296875/ r2스코어 : 0.2930225389373261/ RMSE :  152.65206120371045

3. [1130_mmscaler] *MinMaxScaler(): test_csv파일까지 scale 점  : 1.30점
Epoch 01782: early stopping/ loss :  23474.890625/ r2스코어 : 0.2877970510689205/ RMSE :  153.21517175723991
'''
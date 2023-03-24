from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,LSTM, Conv1D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #(1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #(715, 9) count제외

# print(train_csv.columns)
# print(train_csv.info())
# print(train_csv.describe())
# print(type(train_csv))
# print(train_csv.isnull().sum())

###결측치제거### 
train_csv = train_csv.dropna() 
print(train_csv.isnull().sum())
# print(train_csv.info())
print(train_csv.shape)  #(1328, 10)

###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640874, test_size=0.2
)

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv) 

#reshape
print(x_train.shape, x_test.shape) #(1062, 9) (266, 9)
print(test_csv.shape) #(715, 9)
x_train= x_train.reshape(1062,9,1)
x_test= x_test.reshape(266,9,1)
test_csv = test_csv.reshape(715,9,1)   ###reshape : test_csv 모델에서 돌려서 평가해주니까 reshape해줘야함!!###

#2. 모델구성

model = Sequential()
model.add(Conv1D(16,(2),padding='same',input_shape=(9,1))) 
model.add(Conv1D(filters=5, kernel_size=(2), padding='valid', activation='relu')) 
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
              verbose=1, 
              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1,
          callbacks=[es])
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

#시간저장
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(path_save + 'submit_conv1D_' + date+ '.csv') # 파일생성



'''
5. *RobustScaler
Epoch3000
loss :  1989.625732421875
r2스코어 : 0.7303451862110664
RMSE :  44.605220624545105

*dnn->cnn
loss :  4083.2978515625
r2스코어 : 0.4465889161208677
RMSE :  63.900687329386685

*LSTM
r2스코어 : 0.4558810671385245
RMSE :  63.36194815967994

*Conv1D
loss :  3100.331787109375
r2스코어 : 0.579810698676325
RMSE :  55.680624582263995
'''
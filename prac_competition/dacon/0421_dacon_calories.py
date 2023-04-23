import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터 
path = './_data/dacon_calories/'
path_save = './_save/dacon_calories/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape) 
# print(train_csv.info())


#결측치 확인 
# print(train_csv.isnull().sum()) #결측치 없음
# train_csv = train_csv.dropna()
# print(train_csv.shape, test_csv.shape)  #(7500, 10) (7500, 9)
x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']
x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

#라벨인코더
le1 = LabelEncoder()
le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])
x['Weight_Status'] = le1.fit_transform(x['Weight_Status'])
# train_csv['Gender'] = le1.fit_transform(train_csv['Gender'])
test_csv['Weight_Status'] = le1.fit_transform(test_csv['Weight_Status'])
# test_csv['Gender'] = le1.fit_transform(test_csv['Gender'])
# print(train_csv['Weight_Status'].value_counts())
# print(train_csv['Gender'].value_counts())


#데이터 분리 
# x = train_csv.drop(['Calories_Burned'], axis=1)
# y = train_csv['Calories_Burned']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640874, test_size=0.2)

#스케일링 
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.fit_transform(test_csv)

#2. 모델구성 
model = Sequential()
model.add(Dense(64, activation='selu', input_dim=9))
model.add(BatchNormalization())
model.add(Dense(32, activation='selu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='selu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='selu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='selu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='selu'))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=50, 
                   verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, validation_split=0.2,
          epochs=1000, batch_size=8, verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

#'mse'->rmse로 변경
import numpy as np
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

#5. 제출 
y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit

import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(path_save + 'submit_dacon_cal_' + date + '.csv')
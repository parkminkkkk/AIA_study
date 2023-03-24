'''
#데이터 형식 바꾸는 방법 2가지(pd->np)
print(dataset['T (datasetegC)'])             #데이터 형태 : pandatasetas
print(dataset['T (datasetegC)'].values)      #데이터 형태 : numpy
print(dataset['T (datasetegC)'].to_numpy())  #데이터 형태 : numpy
'''
#[시계열데이터_실습] 
#loss='mse', metrics['mae']
#7:2:1 = train(val):test:predatasetict(->RMSE뺴기) : 성능비교 / 순서대로 자르기 (split에서 셔플하면 안됨 )
#행렬함수 벡터형식, 컬럼형태가 안먹힌다면 한개씩 잘라줘야함 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input,LSTM, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터 
path = './_data/kaggle_jena/'
path_save = './_save/kaggle_jena/'

dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col = 0)
print(dataset)  #[420551 rows x 14 columns]
#[420551 rows x 15 columns] # [date Time] : 데이터x => index임  
#성능향상원할 시 date Time에서 년,월, 일, 시간대로 컬럼 빼낼 수는 있음 
#10분단위인 데이터를 몇개씩 자를지 선택! 


#1-1 데이터 확인 및 결측치 제거 
print(dataset.columns)
print(dataset.info()) #결측치 없음 
print(dataset.describe()) # T(temperature)를 y값으로

print(dataset['T (degC)'].values)  #pandas데이터 형식으로 출력됨  -> numpy로 바꿔줌 
 
# import matplotlib.pyplot as plt   
# plt.plot(dataset['T (degC)'].values)    #numpy데이터 형식으로 바꿔줘야함 
# plt.show()

#1-2 데이터 분리
x = dataset.drop(['T (degC)'], axis=1)
print(x)
y = dataset['T (degC)']
print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=False, train_size=0.7)
x_test, x_pred, y_test, y_pred = train_test_split(x_test, y_test, shuffle=False, train_size=0.67)

print(x_train.shape, y_train.shape) #(294385, 13) #(294385,)
print(x_test.shape, y_test.shape)  #(84531, 13) (84531,)
print(x_pred.shape, y_pred.shape)  #(41635, 13) (41635,)


timesteps = 10          
def split_X(dataset, timesteps):                   
    aaa = []                                      
    for i in range(len(dataset) - timesteps): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)                          

x_trains=split_X(x_train,timesteps)
x_tests=split_X(x_test,timesteps)
x_preds=split_X(x_pred,timesteps)

print(x_trains.shape) #(294375, 10, 13)
print(x_tests.shape)  #(84521, 10, 13)
print(x_preds.shape)  #(41625, 10, 13)

y_trains = y_train[timesteps:]
y_tests = y_test[timesteps:]
y_preds = y_pred[timesteps:]

print(y_trains.shape) #(294375,)

#2. 모델구성 
model = Sequential()
model.add(LSTM(16, input_shape=(10,13))) 
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='loss', patience=10, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_trains, y_trains, epochs=100, callbacks=(es))

#4. 평가, 예측 

loss = model.evaluate(x_tests, y_tests)
print('loss : ', loss)

y_predict = model.predict(x_preds)
# print('predict:', predict)
r2 = r2_score(y_preds, y_predict)
print('r2 : ', r2)


#'mse'->rmse로 변경
def RMSE(x, y): 
    return np.sqrt(mean_squared_error(x,y))

rmse = RMSE(y_preds, y_predict)
print("rmse : ", rmse)

'''
loss :  [2.0559871196746826, 1.0672963857650757]
#y_predict: [[ 4.2016144]... [-4.984675 ]]
RMSE :  1.3203625337053624

loss :  [2.422412395477295, 1.0773078203201294]
#y_predict: [[ 3.7247777]...[-3.1754308]]
RMSE :  1.400186049641971

loss :  [58.66366958618164, 6.303465366363525]
r2 :  -0.12704651187321603
rmse :  8.294033446423985
'''


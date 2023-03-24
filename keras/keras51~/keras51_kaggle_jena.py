'''
#데이터 형식 바꾸는 방법 2가지(pd->np)
print(dataset['T (datasetegC)'])             #데이터 형태 : pandatasetas
print(dataset['T (datasetegC)'].values)      #데이터 형태 : numpy
print(dataset['T (datasetegC)'].to_numpy())  #데이터 형태 : numpy
'''
#[시계열데이터_실습] 
#(참고)keras43_split4_행렬자르기
#loss='mse', metrics['mae']
#7:2:1 = train(val):test:predatasetict(->RMSE뺴기) : 성능비교 / 순서대로 자르기 (split에서 셔플하면 안됨 )
#행렬함수 벡터형식, 컬럼형태가 안먹힌다면 한개씩 잘라줘야함 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,LSTM, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping


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

print(x_train.shape, y_train.shape) #(294385, 13)
print(x_test.shape, y_test.shape)  #(84531, 13)
print(x_pred.shape, y_pred.shape)  #(41635, 13)


timesteps = 100            
def split_X(dataset, timesteps):                   
    aaa = []                                      
    for i in range(len(dataset) - timesteps +1): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)                          






#2. 모델구성 
model = Sequential()
model.add(LSTM(16, input_shape=(100,13))) 
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

model.fit(x_train1, y_train1, epochs=1, callbacks=(es))

#4. 평가, 예측 

loss = model.evaluate(x_test1, y_test1)
x_predict = np.array(x_pred).reshape(-1,13,1)
print('loss : ', loss)

y_predict = model.predict(x_pred1)
print('y_pred:', y_predict)

#'mse'->rmse로 변경
import numpy as np
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_pred1)
print("RMSE : ", rmse)

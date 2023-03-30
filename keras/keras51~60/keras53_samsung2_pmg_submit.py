import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input,LSTM, Conv1D, Reshape, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터 
path = './_data/시험/'
path_save = './_save/samsung/'

datasetS = pd.read_csv(path + '삼성전자 주가3.csv', encoding='cp949', index_col=0)
print(datasetS) #[3260 rows x 17 columns]

datasetH = pd.read_csv(path + '현대자동차2.csv', encoding='cp949', index_col=0)
print(datasetH) #[3140 rows x 17 columns]

# datasetS = datasetS[::-1]
# datasetH = datasetH[::-1] #start:end:-1(역순) 모든 요소 출력
# print(datasetS.head)


#1-1 데이터 확인 및 결측치 제거 
# #삼성
# print(datasetS.columns)
# print(datasetS.info()) #dtypes: float64(3), object(14) #결측치(거래량, 금액(백만), )
# print(datasetS.describe())
# print(type(datasetS))  #<class 'pandas.core.frame.DataFrame'>
datasetS = datasetS.dropna()
print(datasetS.isnull().sum())
# '''
# 거래량 3 / 금액(백만) 3
# '''

# #현대
# print(datasetH.columns)
# print(datasetH.info()) #dtypes: float64(3), object(14) #결측치(거래량, 금액(백만), )
# print(datasetH.describe())
# print(type(datasetH))  #<class 'pandas.core.frame.DataFrame'>
datasetH = datasetH.dropna()
print(datasetH.isnull().sum())


#1-2 데이터 분리 
x1_ss = datasetS.drop(['전일비','등락률','금액(백만)','외인(수량)','외국계','프로그램','외인비'], axis=1)
print(x1_ss)
x2_hd = datasetH.drop(['전일비','등락률','금액(백만)','외인(수량)','외국계','프로그램','외인비'], axis=1)
print(x2_hd)
y1_ss = datasetS['종가']


#X,Y
# x1_ss = x1_ss[:900].values
x1_ss = np.array(x1_ss[:1000])
x2_hd = np.array(x2_hd[:1000])
y1_ss = np.array(y1_ss[:1000])

x1_ss = x1_ss[::-1]
x2_hd = x2_hd[::-1]
y1_ss = y1_ss[::-1]

print(x1_ss.shape, x2_hd.shape) #(1000, 11) (1000, 11)
print(y1_ss.shape) #(1000,)

x1_ss = np.char.replace(x1_ss.astype(str), ',', '').astype(np.float64)
y1_ss = np.char.replace(y1_ss.astype(str), ',', '').astype(np.float64)
x2_hd = np.char.replace(x2_hd.astype(str), ',', '').astype(np.float64)

x1_train, x1_test, x2_train, x2_test, \
y1_train, y1_test= train_test_split(
    x1_ss, x2_hd, y1_ss, shuffle=False, train_size=0.7)

print(x1_train.shape) #(700, 11)

timesteps = 4 

#Scaler
scaler=StandardScaler()
x1_train=scaler.fit_transform(x1_train)
x1_test=scaler.transform(x1_test)
scaler=RobustScaler()
x2_train=scaler.fit_transform(x2_train)
x2_test=scaler.transform(x2_test)

x1_pred = x1_test[-timesteps:].reshape(1,timesteps,9)
x2_pred = x2_test[-timesteps:].reshape(1,timesteps,9)

#split함수정의  
def splitX(dataset, timesteps):                   
    aaa = []                                      
    for i in range(len(dataset) - timesteps): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)      

x1_trains = splitX(x1_train, timesteps)
x1_tests = splitX(x1_test, timesteps)
x2_trains = splitX(x2_train, timesteps)
x2_tests = splitX(x2_test, timesteps)


print(x1_trains.shape, x2_trains.shape)  #(695, 5, 9) (695, 5, 9)
print(x1_tests.shape, x2_tests.shape)    #(295, 5, 9) (295, 5, 9)
print(x1_pred.shape, x2_pred.shape)      #(1, 5, 9) (1, 5, 9)

y1_trains = y1_train[timesteps:]
y1_tests = y1_test[timesteps:]

print(y1_trains.shape, y1_tests.shape)  #(690,) (290,)


#2. 모델구성 

#모델 로드
model = load_model('./_save/samsung/keras53_samsung2_pmg.h5')  #가중치 저장
# model.summary()


#3. 컴파일, 훈련 

# model. compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
#                    verbose=1, 
#                    restore_best_weights=True
#                    )

# model.fit([x1_trains, x2_trains], [y1_trains], epochs=500, batch_size=128, validation_split=0.2,
#           callbacks=[es])

#모델 저장
# model.save('./_save/samsung/keras53_samsung15_pmg.h5')  ##컴파일, 훈련 다음에 save


#4. 평가, 예측 

loss = model.evaluate([x1_tests, x2_tests], y1_tests)
print("loss:", loss)

y_pred = model.predict([x1_pred, x2_pred])
# print(y_pred.shape)
print("내일(0329)종가:", "%.2f"% y_pred[0]) 

'''
#(0329)실제 삼성 종가 :62,700원#
loss: 4579608.0
내일(0329)종가: 63486.34
'''
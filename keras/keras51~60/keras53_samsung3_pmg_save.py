###################################################################################################
#2023-03-27#
#[대회] 삼성전자와 현대 자동차 주가로 삼성전자 주가 맞추기

#[규칙]
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timestep와 feature는 알아서 (자유)
# 제공된 데이터 외 추가 데이터 사용금지 
# 앙상블 필수 포함 (두개의 행의 크기가 같아야 모델 돌아감(1000일치=1000일치))

# 1. 삼성전자 28일(화) 내일(29일) 삼성종가 맞추기 (점수배점0.3점) -다음날 맞추기
# 2. 현대자동차 29일(수) 모레(30일) 현대시가 맞추기 (점수배점0.7점) - 다다음날 맞추기(timestep한칸 띄고..)

'''
마감시간 : 27일(월) 23시 59분 59초 
마감시간 : 28일(화) 23시 59분 59초  
 
(화)메일제목 : 박민경 [삼성 1차] 60,350.07원  #round해야함!! 소수 2자리까지 
(수)메일제목 : 박민경 [현대 2차] 60,350.07원 

2번, 4번파일 첨부
첨부파일 : keras53_samsung2_pmg_submit.py   #가중치 불러오는 파일
첨부파일 : keras53_samsung4_pmg_submit.py

_save/samsung/
가중치 : _save/samsung/keras53_samsung2_pmg.h5/ hdf5 /mcp 모두 가능
가중치 : _save/samsung/keras53_samsung4_pmg.h5/ hdf5
 
'''
###################################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from tensorflow.keras.models import Sequential, Model
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
y2_hd = datasetH['시가']


#X,Y
# x1_ss = x1_ss[:900].values
x1_ss = np.array(x1_ss[:1000])[::-1]
x2_hd = np.array(x2_hd[:1000])[::-1]
y2_hd = np.array(y2_hd[:1000])[::-1]

# x1_ss = x1_ss[::-1]
# x2_hd = x2_hd[::-1]
# y2_hd = y2_hd[::-1]

print(x1_ss.shape, x2_hd.shape) #(1000, 9) (1000, 9)
print(y2_hd.shape) #(1000,)

x1_ss = np.char.replace(x1_ss.astype(str), ',', '').astype(np.float64)
x2_hd = np.char.replace(x2_hd.astype(str), ',', '').astype(np.float64)
y2_hd = np.char.replace(y2_hd.astype(str), ',', '').astype(np.float64)

x1_train, x1_test, x2_train, x2_test, \
y2_train, y2_test= train_test_split(
    x1_ss, x2_hd, y2_hd, shuffle=False, train_size=0.7)

print(x1_train.shape) #(700, 9)

timesteps = 5

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
    for i in range(len(dataset) - timesteps-1): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)      

x1_trains = splitX(x1_train, timesteps)
x1_tests = splitX(x1_test, timesteps)
x2_trains = splitX(x2_train, timesteps)
x2_tests = splitX(x2_test, timesteps)


print(x1_trains.shape, x2_trains.shape)  #(694, 5, 9) (694, 5, 9)
print(x1_tests.shape, x2_tests.shape)    #(294, 5, 9) (294, 5, 9)
print(x1_pred.shape, x2_pred.shape)      #(1, 5, 9) (1, 5, 9)

y2_trains = y2_train[timesteps+1:]
y2_tests = y2_test[timesteps+1:]

print(y2_trains.shape, y2_tests.shape)  #(694,) (294,)


#2. 모델구성 
#2-1. 삼성모델 
input1 = Input(shape=(timesteps,9))
dense1 = LSTM(16, activation='relu',return_sequences=True, name='ss1')(input1)
dense2 = LSTM(32, activation='swish', name='ss2')(dense1)
dense3 = Dense(32, activation='selu', name='ss3')(dense2)
dense4 = Dense(16, activation='selu', name='ss4')(dense3)
output1 = Dense(16, name='output1')(dense4)  


#2-2. 현대모델 
input2 = Input(shape=(5,9))
dense11 = LSTM(16, return_sequences=True, activation='selu', name='hd1')(input2)
dense12 = LSTM(16, activation='relu', name='hd2')(dense11)
dense13 = Dense(32, activation='swish', name='hd3')(dense12)
dense14 = Dense(16, activation='swish', name='hd4')(dense13)
output2 = Dense(16, name='output2')(dense14)

#2-3. 모델 합침 
merge1 = concatenate([output1, output2], name='mg1')  
merge2 = Dense(32, activation='selu', name='mg2')(merge1)
merge3 = Dense(16, activation='swish', name='mg3')(merge2)
merge4 = Dense(32, activation='swish', name='mg4')(merge3)
merge5 = Dense(16, activation='relu', name='mg5')(merge4)
last_output = Dense(1, name='last')(merge5)



#2-6 모델 정의 
model = Model(inputs=[input1, input2], outputs=[last_output])

# model.summary()


#3. 컴파일, 훈련 

model. compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit([x1_trains, x2_trains], [y2_trains], epochs=100, batch_size=8, validation_split=0.2,
          callbacks=[es])


#모델 저장
model.save('./_save/samsung/keras53_samsung40_.h5')  ##컴파일, 훈련 다음에 save


#4. 평가, 예측 

loss = model.evaluate([x1_tests, x2_tests], y2_tests)
print("loss:", loss)

y_pred = model.predict([x1_pred, x2_pred])
# print(y_pred.shape)
print("모레(0330)시가:", "%.2f"% y_pred) 

#그래프
# import matplotlib.pyplot as plt
# plt.plot(range(len(y2_tests)),y2_tests,label='real', color='red')
# plt.plot(range(len(y2_tests)),model.predict([x1_tests,x2_tests]),label='model')
# plt.legend()
# plt.show()

'''
# 데이터 두개 추가 ('./_save/samsung/keras53_samsung40_.h5')
loss: 36684032.0
모레(0330)시가: 172784.67

#('./_save/samsung/keras53_samsung41_pmg.h5')
loss: 89005832.0
모레(0330)시가: 168656.48
=================================================================
# 데이터 두개 삭제 ('./_save/samsung/keras53_samsung41_pmg.h5')
loss: 35186024.0
모레(0330)시가: 175022.05

#('./_save/samsung/keras53_samsung42_pmg.h5')
loss: 60284364.0
모레(0330)시가: 177634.88

'''

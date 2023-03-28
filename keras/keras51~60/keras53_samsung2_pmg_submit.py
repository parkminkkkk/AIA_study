###################################################################################################
#2023-03-27#
#[대회] 삼성전자와 현대 자동차 주가로 삼성전자 주가 맞추기

#[규칙]
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timestep와 feature는 알아서 (자유)
# 제공된 데이터 외 추가 데이터 사용금지 
# 앙상블 필수 포함 (두개의 행의 크기가 같아야 모델 돌아감(1000일치=1000일치))

# 1. 삼성전자 28일(화) 종가 맞추기 (점수배점0.3점) -다음날 맞추기
# 2. 삼성전자 29일(수) 아침 시가 맞추기 (점수배점0.7점) - 다다음날 맞추기(timestep한칸 띄고..)

'''
마감시간 : 27일(월) 23시 59분 59초 
마감시간 : 28일(화) 23시 59분 59초  
 
(화)메일제목 : 박민경 [삼성 1차] 60,350.07원  #round해야함!! 소수 2자리까지 
(수)메일제목 : 박민경 [삼성 2차] 60,350.07원 

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input,LSTM, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터 
path = './_data/시험/'
path_save = './_save/samsung/'

datasetS = pd.read_csv(path + '삼성전자 주가2.csv', encoding='cp949', index_col=0)
print(datasetS) #[3260 rows x 17 columns]

datasetH = pd.read_csv(path + '현대자동차.csv', encoding='cp949', index_col=0)
print(datasetH) #[3140 rows x 17 columns]


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
x1_ss = datasetS.drop(['전일비','외인(수량)','외국계','프로그램','외인비'], axis=1)
print(x1_ss)
x2_hd = datasetH.drop(['전일비','외인(수량)','외국계','프로그램','외인비'], axis=1)
print(x2_hd)
y1_ss = datasetS['종가']



#X,Y
x1_ss = np.array(x1_ss[:1000])
x2_hd = np.array(x2_hd[:1000])

y1_ss = np.array(y1_ss[:1000])


print(x1_ss.shape, x2_hd.shape) #(1000, 11) (1000, 11)
print(y1_ss.shape) #(1000,)

x1_ss = np.char.replace(x1_ss.astype(str), ',', '').astype(np.float64)
y1_ss = np.char.replace(y1_ss.astype(str), ',', '').astype(np.float64)
x2_hd = np.char.replace(x2_hd.astype(str), ',', '').astype(np.float64)

x1_train, x1_test, x2_train, x2_test, \
y1_train, y1_test= train_test_split(
    x1_ss, x2_hd, y1_ss, shuffle=False, train_size=0.7)

print(x1_train.shape) #(700, 11)


scaler=StandardScaler()
x1_train=scaler.fit_transform(x1_train)
x1_test=scaler.transform(x1_test)
x2_train=scaler.transform(x2_train)
x2_test=scaler.transform(x2_test)


timesteps = 10          
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

print(x1_trains.shape, x2_trains.shape)  #(690, 10, 11) (690, 10, 11)
print(x1_tests.shape, x2_tests.shape)    #(290, 10, 11) (290, 10, 11)

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

# model.fit([x1_trains, x2_trains], [y1_trains], epochs=300, batch_size=16, validation_split=0.2,
#           callbacks=[es])


# #모델 저장
# model.save('./_save/samsung/keras53_samsung21_pmg.h5')  ##컴파일, 훈련 다음에 save


#4. 평가, 예측 

loss = model.evaluate([x1_tests, x2_tests], y1_tests)
print("loss:", loss)

y_pred = model.predict([x1_tests, x2_tests])
# print(y_pred.shape)
print("23.03.28의 종가:", y_pred)
print("23.03.28의 종가:", y_pred[0])


'''
23.03.28의 종가: [62967.12]
'''
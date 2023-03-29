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
x1_ss = datasetS.drop(['전일비','등락률','금액(백만)','신용비','기관','외인(수량)','외국계','프로그램','외인비'], axis=1)
print(x1_ss)
x2_hd = datasetH.drop(['전일비','등락률','금액(백만)','신용비','기관','외인(수량)','외국계','프로그램','외인비'], axis=1)
print(x2_hd)
y2_hd = datasetH['시가']


#X,Y
# x1_ss = x1_ss[:900].values
x1_ss = np.array(x1_ss[:300])[::-1]
x2_hd = np.array(x2_hd[:300])[::-1]
y2_hd = np.array(y2_hd[:300])[::-1]

# x1_ss = x1_ss[::-1]
# x2_hd = x2_hd[::-1]
# y2_hd = y2_hd[::-1]

print(x1_ss.shape, x2_hd.shape) #(1000, 11) (1000, 11)
print(y2_hd.shape) #(1000,)

x1_ss = np.char.replace(x1_ss.astype(str), ',', '').astype(np.float64)
x2_hd = np.char.replace(x2_hd.astype(str), ',', '').astype(np.float64)
y2_hd = np.char.replace(y2_hd.astype(str), ',', '').astype(np.float64)

x1_train, x1_test, x2_train, x2_test, \
y2_train, y2_test= train_test_split(
    x1_ss, x2_hd, y2_hd, shuffle=False, train_size=0.7)

print(x1_train.shape) #(700, 11)

timesteps = 5

#Scaler
scaler=StandardScaler()
x1_train=scaler.fit_transform(x1_train)
x1_test=scaler.transform(x1_test)
scaler=RobustScaler()
x2_train=scaler.fit_transform(x2_train)
x2_test=scaler.transform(x2_test)


#split함수정의  
def splitX(dataset, timesteps):                   
    aaa = []                                      
    for i in range(len(dataset) - timesteps+1): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)      

x1_trains = splitX(x1_train, timesteps)
x1_tests = splitX(x1_test, timesteps)
x1_pred = np.reshape(x1_tests[-1],([1]+list(x1_trains.shape[1:])))
x1_tests = x1_tests[:-2]

x2_trains = splitX(x2_train, timesteps)
x2_tests = splitX(x2_test, timesteps)
x2_pred = np.reshape(x2_tests[-1],([1]+list(x2_trains.shape[1:])))
x2_tests = x2_tests[:-2]


print(x1_trains.shape, x2_trains.shape)  #(696, 5, 9) (696, 5, 9)
print(x1_tests.shape, x2_tests.shape)    #(294, 5, 9) (294, 5, 9)
print(x1_pred.shape, x2_pred.shape)      #(1, 5, 9) (1, 5, 9)

y2_trains = np.concatenate((y2_train[timesteps+1:],y2_test[:2]),axis=0)    #x,y데이터 개수 차이만큼 2개 데이터 생성 (y2_train에 y2_test데이터 값 2개 붙임)
y2_tests = y2_test[timesteps+1:]

print(y2_trains.shape, y2_tests.shape)  #(696,) (294,)


#2. 모델구성 

#모델 로드
# model.summary()
model = load_model('./_save/samsung/keras53_samsung4_pmg.h5')  #가중치 저장


# #3. 컴파일, 훈련 

# model. compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', patience=32, mode='min', 
#                    verbose=1, 
#                    restore_best_weights=True
#                    )

# hist=model.fit([x1_trains, x2_trains], [y2_trains], epochs=256, batch_size=8, validation_split=0.2,
#           callbacks=[es])




#4. 평가, 예측 

loss = model.evaluate([x1_tests, x2_tests], y2_tests)
print("loss:", loss)

y_pred = model.predict([x1_pred, x2_pred])
# print(y_pred.shape)
# print("모레(0330)시가:", np.round(y_pred,2)) 

print("모레(0330)시가:", "%.2f"% y_pred) 

#그래프
# import matplotlib.pyplot as plt
# plt.plot(range(len(y2_tests)),y2_tests,label='real', color='red')
# plt.plot(range(len(y2_tests)),model.predict([x1_tests,x2_tests]),label='model')
# plt.legend()
# plt.show()

'''

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
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
y2_hd = datasetH['종가']


#X,Y
x1_ss = np.array(x1_ss[:1000])
x2_hd = np.array(x2_hd[:1000])

y1_ss = np.array(y1_ss[:1000])
y2_hd = np.array(y2_hd[:1000])

print(x1_ss.shape, x2_hd.shape) #(1000, 11) (1000, 11)
print(y1_ss.shape, y2_hd.shape) #(1000,) (1000,)

x1_ss = np.char.replace(x1_ss.astype(str), ',', '').astype(np.float64)
y1_ss = np.char.replace(y1_ss.astype(str), ',', '').astype(np.float64)
x2_hd = np.char.replace(x2_hd.astype(str), ',', '').astype(np.float64)
y2_hd = np.char.replace(y1_ss.astype(str), ',', '').astype(np.float64)


timesteps = 20          
def splitX(dataset, timesteps):                   
    aaa = []                                      
    for i in range(len(dataset) - timesteps): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)      

x1_ss = splitX(x1_ss, timesteps)
x2_hd = splitX(x2_hd, timesteps)

print(x1_ss.shape)  #(990, 20, 11)
print(x2_hd.shape)    

y1_ss = y1_ss[timesteps:]
y2_hd = y2_hd[timesteps:]


print(y1_ss.shape, y2_hd.shape)   #(980,)

#split
x1_train, x1_test, y1_train, y1_test= train_test_split(
    x1_ss, y1_ss, shuffle=True, random_state=640, train_size=0.7)

x2_train, x2_test, y2_train, y2_test= train_test_split(
    x2_hd, y2_hd, shuffle=True, random_state=640, train_size=0.7)


x1_train= x1_train.reshape(-1,20*11)
x1_test= x1_test.reshape(-1,20*11)
x2_train= x2_train.reshape(-1,20*11)
x2_test= x2_test.reshape(-1,20*11)

scaler=StandardScaler()
x1_train=scaler.fit_transform(x1_train)
x1_test=scaler.transform(x1_test)
x2_train=scaler.transform(x2_train)
x2_test=scaler.transform(x2_test)

x1_train= x1_train.reshape(-1,20,11)
x1_test= x1_test.reshape(-1,20,11)
x2_train= x2_train.reshape(-1,20,11)
x2_test= x2_test.reshape(-1,20,11)


#2. 모델구성 
#2-1. 삼성모델 
input1 = Input(shape=(20,11))
dense1 = LSTM(16, activation='relu', name='ss1')(input1)
dense2 = Dense(32, activation='relu', name='ss2')(dense1)
dense3 = Dense(32, activation='relu', name='ss3')(dense2)
output1 = Dense(16, activation='relu', name='output1')(dense3)  


#2-2. 현대모델 
input2 = Input(shape=(20,11))
dense11 = LSTM(16, activation='relu', name='hd1')(input2)
dense12 = Dense(16, activation='relu', name='hd2')(dense11)
dense14 = Dense(16, activation='swish', name='hd4')(dense12)
output2 = Dense(16, name='output2')(dense14)

#2-3. 모델 합침 
merge1 = concatenate([output1, output2], name='mg1')  
merge2 = Dense(32, activation='selu', name='mg2')(merge1)
merge3 = Dense(32, activation='swish', name='mg3')(merge2)
merge4 = Dense(16, activation='relu', name='mg4')(merge3)
last_output = Dense(1, name='last')(merge3)



#2-6 모델 정의 
model = Model(inputs=[input1, input2], outputs=[last_output])

# model.summary()


#3. 컴파일, 훈련 

model. compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit([x1_train, x2_train], y1_train, epochs=300, batch_size=128, validation_split=0.2,
          callbacks=[es])

#시간저장
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

#모델 저장
model.save('./_save/samsung/keras53_samsung2T_pmg.h5')  ##컴파일, 훈련 다음에 save


#4. 평가, 예측 

loss = model.evaluate([x1_test, x2_test], y1_test)
print("loss:", loss)

y1_pred = model.predict([x1_test, x2_test])
# print(y_pred.shape)
print("23.03.28의 종가:", y1_pred)
print("23.03.28의 종가:", y1_pred[0])

'''
23.03.28의 종가: [62967.12]

#shuffle=F
23.03.28의 종가: [53163.652]
#shuffle=T
23.03.28의 종가: [54994.56]
'''
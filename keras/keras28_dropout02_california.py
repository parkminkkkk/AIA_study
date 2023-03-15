#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=123)

#data scaling(스케일링)
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성 (함수형모델) 
'''
input1 = Input(shape=(8,), name='h1') 
dense1 = Dense(4, name='h2', activation='relu')(input1) 
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(6, name='h3', activation='relu')(drop1)  
drop2 = Dropout(0.2)(dense2)                                                          
dense3 = Dense(4, name='h4',activation='relu')(drop2)     
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(2, name='h5',activation='relu')(drop3)                                                               
output1 = Dense(1, name='h5',)(dense4)
model = Model(inputs=input1, outputs=output1) 
'''
model = Sequential()
model.add(Dense(8, input_shape=(8,)))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

#시간저장
import datetime 
date = datetime.datetime.now()  #현재시간 데이터에 넣어줌
print(date)  #2023-03-14 11:15:21.154501
date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute
#시간을 문자데이터로 바꿈 : 문자로 바꿔야 파일명에 넣을 수 있음 
print(date) #0314_1115

#경로명 
filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04 : 4번째자리, .4: 소수점자리 / hist에서 가져옴 



from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=10000, validation_split=0.2, batch_size=64, validation_split=0.2, verbose=1,
          callbacks=(es)) #[mcp])



#4. 평가, 예측 
from sklearn.metrics import r2_score
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)

'''
1. *MinMaxScaler()
Epoch 00258: early stopping/ loss : 0.3793175220489502 /r2스코어 :  0.7147589621125761

2. *dropout 
loss : 0.4030558168888092
r2스코어 0.696908172725553
'''
#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=123)

#data scaling(스케일링)
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

#2. 모델구성 (함수형모델) 

# input1 = Input(shape=(10, ))
# dense1 = Dense(32, activation='relu')(input1)
# drop1 = Dropout(0.1)(dense1)
# dense2 = Dense(16, activation='relu')(drop1)
# drop2 = Dropout(0.1)(dense2)
# dense3 = Dense(8, activation='relu')(drop2)
# drop3 = Dropout(0.1)(dense3)
# dense4 = Dense(2, activation='relu')(drop3)
# output1 = Dense(1, activation='linear')(dense4)
# model = Model(inputs=input1, outputs=output1)

model = Sequential()
model.add(Dense(32, activation='linear', input_shape=(10, )))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(16, Dropout(0.1)))
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
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04 : 4번째자리, .4: 소수점자리 - hist에서 가져옴 



from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=5000, batch_size=100, validation_split=0.2, 
          callbacks=(es)) #[mcp])



#4. 평가, 예측 
from sklearn.metrics import r2_score
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)


'''
1.MinMaxScaler(), 
Epoch 00328: early stopping/ loss :  2763.925537109375/ r2스코어 : 0.5612932891200189

2. *dropout 
Epoch 00339: early stopping
loss : 2790.89208984375
r2스코어 0.5570130026895617
'''
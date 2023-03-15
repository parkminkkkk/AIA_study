#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정


from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import pandas as pd

#1. 데이터 
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #(10886, 11)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)

###결측치제거### 
print(train_csv.isnull().sum()) #isnull이 True인것의 합계 : 각 컬럼별로 결측치 몇개인지 알수 있음
#결측치 없음

###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=333)

###data scaling(스케일링)###
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성 (함수형모델) 

'''
input1 = Input(shape=(8,), name='h1') 
dense1 = Dense(8, name='h2')(input1) 
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(4, name='h3', activation='relu')(drop1)  
drop2 = Dropout(0.2)(dense2)                                                          
dense3 = Dense(16, name='h4',activation='relu')(drop2)     
drop3 = Dropout(0.5)(dense3)                                                          
output1 = Dense(1, name='h5',)(drop3)
model = Model(inputs=input1, outputs=output1) 
'''

model = Sequential()
model.add(Dense(8, input_shape=(8,)))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, Dropout(0.1)))
# model.add(Dense(16, activation='relu', Dropout(0.1)))
model.add(Dense(4, activation='relu'))
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
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, 
          callbacks=(es)) #[mcp])



#4. 평가, 예측 
from sklearn.metrics import r2_score, mean_squared_error
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)


#'mse'->rmse로 변경

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

#submission.csv 만들기 
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0314_1500_dropout.csv') # 파일생성

'''
3. [1130_mmscaler] *MinMaxScaler(): test_csv파일까지 scale 점  : 1.30점
Epoch 01782: early stopping/ loss :  23474.890625/ r2스코어 : 0.2877970510689205/ RMSE :  153.21517175723991

4. [1500_dropout] *dropout 
loss : 23064.154296875
r2스코어 0.26201214728137556
RMSE :  151.8688661100168
'''
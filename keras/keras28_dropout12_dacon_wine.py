#Dancon_wine 
#결측치/ 원핫인코딩, 데이터분리, 스케일링/ 함수형,dropout
#다중분류 - softmax, categorical

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

#1. 데이터 
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#1-1 결측치 제거 
# print(train_csv.isnull().sum())
# print(train_csv.info())
# train_csv = train_csv.dropna() #결측치없음 

x = train_csv.drop(['quality', 'type'], axis=1)
print(x.shape)                       #(5497, 11)
y = train_csv['quality']
print(type(y))
print(y)
print("y_shape:", y.shape)           #(5497,)
print('y의 라벨값 :', np.unique(y))  #[3 4 5 6 7 8 9]
test_csv = test_csv.drop(['type'], axis=1)

#1-2 one-hot-encoding
print('y의 라벨값 :', np.unique(y))  #[3 4 5 6 7 8 9]
import pandas as pd
y=pd.get_dummies(y)
y = np.array(y)
print(y.shape)                       #(5497, 7)

###########################################################################
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# print(y.shape)
# print(type(y))
# y = train_csv['quality'].values 
# y = y.reshape(-1,1)
# print(y.shape)
# print(type(y)) 
# y = ohe.fit_transform(y).toarray()
# print(y.shape) 

# print('y의 라벨값 :', np.unique(y))  #[3 4 5 6 7 8 9]
# import pandas as pd
# y=pd.get_dummies(y)
# y = np.array(y)
# print(y.shape)                       #(5497, 7)

# keras to_categorical
#import numpy as np
#from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) 
#print(y.shape)   #(5497, 10)
# y = np.delet(y, 0, axis=1) #앞에 0,1,2열 없에줘야함 x3번 반복 
# y = y[: , 3:]  #행은 0-0, 열은 3-끝까지
############################################################################

#1-3 데이터분리 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=640874)

#1-4 스케일링 
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
# print(np.min(x_test), np.max(x_test)) 

#2. 모델구성 
input1 = Input(shape=(11,))
dense1 = Dense(32,activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(4, activation='relu')(drop3)
output1 = Dense(7, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(8, activation='relu', input_shape=(11,)))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


#시간저장
import datetime 
date = datetime.datetime.now()  #현재시간 데이터에 넣어줌
print(date)  #2023-03-14 11:15:21.154501
date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute
#시간을 문자데이터로 바꿈 : 문자로 바꿔야 파일명에 넣을 수 있음 
print(date) #0314_1115

#경로명 
filepath = './_save/MCP/keras28_12/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04 : 4번째자리, .4: 소수점자리 - hist에서 가져옴 


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_split=0.1, verbose=1, 
          callbacks=[es, mcp])
  
#4. 평가예측 
results = model.evaluate(x_test, y_test)
print('results:', results)  
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# print(y_pred.shape)
# print(y_test)
# print(y_test.shape)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score:', acc)


#submission.csv 만들기 
y_submit = model.predict(test_csv)
# print(y_submit)

y_submit = np.argmax(y_submit, axis=1)
# print(y_submit.shape)
y_submit += 3

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = y_submit
# print(submission)
submission.to_csv(path_save + 'submit_0314_1805_MM.csv') # 파일생성

'''
*MM
results: [1.0840587615966797, 0.5518181920051575]
accuracy_score: 0.5518181818181818
*MM
results: [1.034911036491394, 0.5600000023841858]
accuracy_score: 0.56
*MM
results: [0.9965323209762573, 0.5727272629737854]
accuracy_score: 0.5727272727272728
*MM- Dense,dropout 추가
results: [0.9915247559547424, 0.5754545331001282]
accuracy_score: 0.5754545454545454
'''
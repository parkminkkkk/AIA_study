#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=777)

#data scaling(스케일링)
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성 (함수형모델) 
'''
model = Sequential()
model.add(Dense(8, input_shape=(8,)))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, Dropout(0.1)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
'''

input1 = Input(shape=(30, ))
dense1 = Dense(16, activation='linear')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(8, activation='relu')(drop1)
dense3 = Dense(4, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
drop2 = Dropout(0.2)(dense4)
dense5 = Dense(4, activation='relu')(drop2)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse'])

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
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=5000, validation_split=0.2, batch_size=32,
          callbacks=(es)) #[mcp])



#4. 평가, 예측 
from sklearn.metrics import r2_score, accuracy_score
results = model.evaluate(x_test, y_test, verbose=0)
print('results:', results) 

y_predict = np.round(model.predict(x_test))
# print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)


'''
4.*MinMaxScaler
Epoch 00248: early stopping
results: [0.18867246806621552, 0.9473684430122375, 0.0516931377351284]
acc:  0.9473684210526315

5.*dropout
Epoch 00137: early stopping
results: [0.09837757050991058, 0.9649122953414917, 0.02767413668334484]
acc:  0.9649122807017544
'''
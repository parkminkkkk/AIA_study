import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

#1. 데이터 

#1-1 데이터 가져오기
path = 'd:/study/_data/dacon_wine/'
path_save = './_save/dacon_wine/'


train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_csv.isnull().sum()) 결측치 없음

#1-2 데이터 정하기
x = train_csv.drop(['quality', 'type'], axis=1) #(5497, 11)
y = train_csv['quality']  #(5497,)
test_csv = test_csv.drop(['type'], axis=1)

#1-3 onehotencoding
print(np.unique(y))  #[3 4 5 6 7 8 9]
y=pd.get_dummies(y)
y = np.array(y)
print(y.shape)   #(5497, 7)

#1-4 데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=640874)

#1-5 스케일링
scaler = MaxAbsScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 


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

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_split=0.1, verbose=1, 
          callbacks=(es)) #[mcp])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('results:', results)  
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

#5. 파일 생성 

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
y_submit += 3

submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submit_csv['quality'] = y_submit
# print(y_submit)

submit_csv.to_csv(path_save + 'submit_0314_1810_MA.csv') # 파일생성

'''
*MAbs 
results: [0.9848098754882812, 0.578181803226471]
acc : 0.5781818181818181
'''
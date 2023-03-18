#이진분류 

import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터 
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
print(train_csv)  
# [652 rows x 9 columns] #(652,9)

test_csv= pd.read_csv(path+'test.csv', index_col=0)
print(test_csv) 
#(116,8) #outcome제외

# print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['Outcome'], axis=1)
# print(x)
y = train_csv['Outcome']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, test_size=0.2,
    stratify=y
)

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
#test_csv파일또한 scaler해줘야함! 아니면 제출했을때 점수이상하게 나옴 (train파일을  scale한만큼, test파일도 scale해줘야함)
#train_csv파일에서 x_train,x_test값 가져온것이기 때문에, test_csv파일 scale해줘야함 


#2. 모델구성
# model = Sequential()
# model.add(Dense(8, activation='linear', input_dim=8))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) #이진분류 - 마지막 아웃풋레이어, 'sigmoid'사용 (0,1사이로 한정) 

input1 = Input(shape=(8, ))
dense1 = Dense(8, activation='linear')(input1)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(4, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',   #이진분류 - loss, 'binary_crossentropy'사용  
              metrics=['acc','mse'] 
              ) 
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min', 
              verbose=1,
              restore_best_weights=True
             )  

hist = model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.1,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results) 
 
y_predict = np.round(model.predict(x_test)) #np.round:반올림 / 예측값을 반올림해서 0,1의 값이 나올 수 있도록해줌 (0.5까지는 0으로, 0.6부터는 1로)

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

#submission.csv생성
y_submit = np.round(model.predict(test_csv))  #np.round y_submit에도 해야함!!****
# print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0313_1500_model.csv') # 파일생성


'''
5. [1410]
results: [0.6158730387687683, 0.7022900581359863, 0.2031799852848053]
acc:  0.7022900763358778
6. [1420]/ 0.81점***2등
*stratify=y / seed:640, Dense(8,4,8,4,1), batch_size=32
results: [0.7886713743209839, 0.7175572514533997, 0.19658119976520538]
acc:  0.71

8. *MinMaxScaler : test_csv파일 scale  
Epoch 00315: early stopping : 0.8103448276점
results: [0.5417091846466064, 0.7175572514533997, 0.18293000757694244]
acc:  0.7175572519083969
9. *MinMaxScaler : test_csv파일 scale  : 0.79점
Epoch 00124: early stopping
results: [0.6271610856056213, 0.732824444770813, 0.1999116986989975]
acc:  0.732824427480916

10.* RobustScaler
Epoch 00423: early stopping
results: [0.5853256583213806, 0.7633587718009949, 0.17271968722343445]
acc:  0.7633587786259542

'''

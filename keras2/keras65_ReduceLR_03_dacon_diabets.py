import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터 
path = 'd:/study/_data/dacon_diabetes/'
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

#2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_shape = (8,)))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련 
# model.compile(loss = "mse", optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate= learning_rate)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20, mode = 'min', verbose=1,)
rlr = ReduceLROnPlateau(monitor='val_loss', patience = 10, mode ='auto', verbose=1, factor=0.5)   #es, rlr의 patience는 따로 준다

model.fit(x_train, y_train, epochs =1000, batch_size=32, verbose=1, validation_split=0.2,
            callbacks = [es, rlr])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)

print("loss:", results)

'''
Epoch 00067: ReduceLROnPlateau reducing learning rate to 0.02500000037252903.
13/13 [==============================] - 0s 16ms/step - loss: 0.4752 - val_loss: 0.5905 - lr: 0.0500
Epoch 00067: early stopping
5/5 [==============================] - 0s 5ms/step - loss: 0.5559
loss: 0.5559444427490234
'''
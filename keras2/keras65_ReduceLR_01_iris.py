import numpy as np 
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=337, shuffle=True
)

#2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련 
# model.compile(loss = "mse", optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.optimizers import Adam
learning_rate = 0.1                                                                               #초반에는 큰 값으로 시작 
optimizer = Adam(learning_rate= learning_rate)
model.compile(loss = 'mse', optimizer = optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20, mode = 'min', verbose=1,)                   #optimizer, rlr 통상 같이 사용해야 함 
rlr = ReduceLROnPlateau(monitor='val_loss', patience = 10, mode ='auto', verbose=1, factor=0.5)   #es, rlr의 patience는 따로 준다// 

model.fit(x_train, y_train, epochs =1000, batch_size=32, verbose=1, validation_split=0.2,
            callbacks = [es, rlr])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)

print("lr:", learning_rate, "loss:", results)



# Epoch 00059: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
# 413/413 [==============================] - 3s 7ms/step - loss: 0.6991 - val_loss: 0.7814 - lr: 0.0025
# Epoch 00059: early stopping
# 129/129 [==============================] - 1s 3ms/step - loss: 0.7079
# loss: 0.7079218626022339
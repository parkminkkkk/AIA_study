from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


#####[실습]#####
'''
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)
#reshape의 주의점 
#중요**데이터의 구조만 바꾸는 것이지, 안의 데이터의 값이나 순서는 바뀌지 않는다**
#(60000,28,14,2)로도 가능 / 이또한, 데이터의 값과 순서는 바꾸지 않음 
#(60000,28*28) (60000,784) 2차원으로 바꿔줌(Dense이용할때) / 데이터와 순서 바꾸지 않음 
#구조(공간,shpae)의 크기는 같기만 하면 됨!! => 28,28,1을 건들일 수 있음(곱하기,나누기했을때 전체값과 동일해야한다)
#cf)transform : 행과 열을 바꿔줌 -> 전혀 다른 값 나옴 
'''

print(np.unique(y_train,return_counts=True)) 
#np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)

#1-1스케일링 
#scaler : 0~255사이 => MinMax가 가장 괜찮/ 255로 나누는 경우도 있음 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#2. 모델구성 
model = Sequential()
model.add(LSTM(16, input_shape=(28,28), activation='linear')) #[batch, / timesteps, feature]   
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

# 함수형
# input1 = Input(shape=(13,1))
# dense1 = LSTM(32, activation='linear')(input1)
# dense2 = Dense(16, activation='relu')(dense1)
# dense3 = Dense(8, activation='relu')(dense2)
# dense4 = Dense(2)(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=10, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=100, batch_size=516, validation_split=0.2, 
          callbacks=(es))

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

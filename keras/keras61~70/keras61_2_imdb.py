from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape, Embedding


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

print(x_train)
print(y_train)
print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(np.unique(y_train, return_counts=True)) #[0 1] #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
###numpy : np.unique / pandas : value_counts : y 라벨값 확인 ###
print(pd.value_counts(y_train)) #1  12500/ 0  12500
# print(pd.unique(y_train))       #[1 0]

print("영화평의 최대길이:", max(len(i) for i in x_train)) #영화평의 최대길이: 2494
print("영화평의 평균길이:", sum(map(len, x_train))/ len(x_train)) #영화평의 평균길이: 238.71364 #중요도를 모르니까 평균치 알아보기 위해서(추측)

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding="pre", maxlen=200,
                        truncating='pre' #maxlen크기보다 더 많은 데이터가 있는 경우는 어디를 자를 것인가 
                        ) 

x_test = pad_sequences(x_test, padding="pre", maxlen=200,
                        truncating='pre' #maxlen크기보다 더 많은 데이터가 있는 경우는 어디를 자를 것인가 
                        ) 

print(x_train.shape, x_test.shape) #(25000, 200) (25000, 200)
x_train=x_train.reshape(-1,200,1)
x_test=x_test.reshape(-1,200,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

#2.모델 구성
model = Sequential()
model.add(Embedding(10000, 40, input_length=200))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax')) #긍정/부정 : 이진분류

#3.컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=50, batch_size=8)
#4.평가 
acc = model.evaluate(x_test,y_test)[1]
print(acc)


'''

'''
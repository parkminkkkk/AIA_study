from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape, Embedding


#1. 데이터 
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2    
    #num_words:어휘에 포함할 최대 단어 수를 지정(최상위 n개로만 구성된 것만 나옴) /#train, test비율 조절 가능
)   
#Embedding의 input_dim = num_words(최대 단어 개수:10000)
#Embedding의 input_length : 지정한 최대길이로 맞춤 (len)
#y는 46개의 클래스파이어(output부분) -> softmax로 맞춤

print(x_train)
print(y_train) #[ 3  4  3 ... 25  3 25]
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape) #(2246,) (2246,)

print(len(x_train[0]), len(x_train[1])) #87 56 : 두 개 길이 다름 (np안에 list형식으로 들어가있음)-> 나중에 길이 맞춰줘야함(최대길이에 맞추기or너무 길면 자르기)
print(np.unique(y_train)) #y는 46개의 클래스파이어(output부분)
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))             #<class 'list'>


print("뉴스기사의 최대길이:", max(len(i) for i in x_train)) #뉴스기사의 최대길이: 2376 #처음부터 끝까지의 데이터 중 최대 길이를 찾기위함
print("뉴스기사의 평균길이:", sum(map(len, x_train))/ len(x_train)) #뉴스기사의 평균길이: 145.5398574927633 #중요도를 모르니까 평균치 알아보기 위해서(추측)

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding="pre", maxlen=100,
                        truncating='pre' #maxlen크기보다 더 많은 데이터가 있는 경우는 어디를 자를 것인가 
                        ) 
print(x_train.shape) #(8982, 100)

x_test = pad_sequences(x_test, padding="pre", maxlen=100,
                        truncating='pre' #maxlen크기보다 더 많은 데이터가 있는 경우는 어디를 자를 것인가 
                        ) 

print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
x_train=x_train.reshape(-1,100,1)
x_test=x_test.reshape(-1,100,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

#2.모델 구성
model = Sequential()
model.add(Embedding(10000, 40, input_length=100))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(46, activation='softmax')) 


#3.컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=50, batch_size=8)
#4.평가 
acc = model.evaluate(x_test,y_test)[1]
print(acc)

'''
loss: 4.5164 - acc: 0.6398
'''
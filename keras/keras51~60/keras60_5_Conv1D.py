#model.add(Flatten())

from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

#1. 데이터 
docs = ['너무 재밌어요', '참 최고에요','참 잘 만든 영화예요', '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요',
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요',
        '환희가 잘 생기긴 했어요', '환희가 안해요']


#긍정 1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])   #y값

#Tokenizer
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밌어요': 5, '최고에요': 6, '만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화
# 입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글쎄요': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20, '
# 연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밌네요': 25, '생기긴': 26, '했어요': 27, '안해요': 28}
print(token.word_counts)
# ([('너무', 2), ('재밌어요', 1), ('참', 3), ('최고에요', 1), ('잘', 2), ('만든', 1), ('영화예요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), (' 
# 보고', 1), ('싶네요', 1), ('글쎄요', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1), ('재미없
# 어요', 1), ('재미없다', 1), ('재밌네요', 1), ('환희가', 2), ('생기긴', 1), ('했어요', 1), ('안해요', 1)])


#수치화
x = token.texts_to_sequences(docs)
print(x) 
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], 
# [21, 22], [23], [2, 24], [1, 25], [4, 3, 26,27], [4, 28]]   => x값(단어사전)

#*Think*#
#(x값)[2,5] = (y값)1/ [1,6]=1 ...
#문제) x값의 크가가 다 다름 -> 빈 곳은 0으로 채우기 (앞에 0채우고, 뒤에 수치 : 뒤쪽에 중요한 어순이 가도록 ex)[0,0,2,5])


#순서에 패딩 채우기
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=6) #'pre' :앞에서부터 / 'post':뒤에서부터  # maxlen : 가장 큰 값(긴 토큰)에 맞춤, 더 작을 경우 잘려나감
print(pad_x)
'''
[[ 0  0  0  2  5]
 [ 0  0  0  1  6]
 [ 0  1  3  7  8]
 [ 0  0  9 10 11]
 [12 13 14 15 16]
 [ 0  0  0  0 17]
 [ 0  0  0  0 18]
 [ 0  0  0 19 20]
 [ 0  0  0 21 22]
 [ 0  0  0  0 23]
 [ 0  0  0  2 24]
 [ 0  0  0  1 25]
 [ 0  4  3 26 27]
 [ 0  0  0  4 28]]
'''
print(pad_x.shape) #(14, 5)  #(데이터개수, maxlen)

word_size = len(token.word_index) #단어사전개수
print("단어사전개수:", word_size)  #단어사전개수: 28


#2. 모델구성 
#뒤쪽 값이 영향을 미치는 경우(어순, 순서) -> 시계열 모델(RNN) 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape, Embedding, Flatten, Conv1D

pad_x = pad_x.reshape(14,6,1)
# pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)

model = Sequential()
model.add(Embedding(input_dim=31, output_dim=32, input_length=6)) #(단어사전개수(최적), output(튜닝), 텍스트최대길이(timesteps, maxlen=5))  
# model.add(Embedding(28,32)) #Error : (None,None) :Flatten바로 못붙임 -> input_length명시
model.add(Conv1D(32,2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #긍정/부정 : 이진분류

# model.summary()

#3. 컴파일, 훈련 
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=50, batch_size=8)


#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc:", acc)

#[실습]# 긍정인지 부정인지 맞추기 
x_predict = ['나는 성호가 정말 재미없다 너무 정말']
token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
x_predict = np.array(x_predict).reshape(-1,6,1)
# x_predict = x_predict.reshape(-1,6,1)
# print(token.word_index)
# pad_x = pad_sequences(x, padding='pre', maxlen=6)

predict = model.predict([x_predict])
print("긍정/부정", predict)



'''
acc: 1.0
긍정/부정 [[0.822127]]
'''

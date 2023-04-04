#자연어처리
#tokenizer : 띄어쓰기 단위별로 토큰화해서 자르겠다. (문장자르기->어절 단위->수치화)
#영어는 tokenizer좋음/ 한국어는 조사때문에 어려움 있음

from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text]) #2개 이상은 리스트 

print(token.word_index) #개수 가장 많은 것에 index 첫번째로 줌, 개수 같을 경우는 순서대로 indexing
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
print(token.word_counts) #단어 사용한 횟수 
#OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1),('엄청', 1), ('마구', 3), ('먹었다', 1)])

#숫자로 바꾸는 작업
x = token.texts_to_sequences([text])
print(x) #[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] #1행 11열 
print(type(x))


#원핫인코딩 (숫자가 커질 수록 가치가 부여되는 것이 아니므로)#

######1. to_categorical ######
from tensorflow.keras.utils import to_categorical
import numpy as np

x = to_categorical(x)
x=np.delete(x, 0, axis=1)
print(x)
print(x.shape) #(1, 11, 9) 
# 문제 발생 : tokenizer로 변화했을때는 1~8까지 (숫자1부터시작), 그러나 to_categorical의 경우(0부터 시작)=> '0'삭제 & reshape(11,8)


######2. get_dummies (1차원 받아들임)######
import pandas as pd
import numpy as np
#1. 
x = pd.get_dummies(np.array(x).reshape(11,))
#2. 
x = pd.get_dummies(np.array(x).ravel()) #flatten과 동일함 
#3.
x = np.array(pd.get_dummies(x[0]))
print(x) 
print(x.shape)
#x = pd.get_dummies(x)/ TypeError: unhashable type: 'list' : 현재 x가 list형태임 
#1)list-> numpy로 바꾸기 #2)왜 list를 받아들이지 못하는가

# x = np.array(x)
# x = x.reshape(-1)
# x = pd.get_dummies(x)
# x = np.array(x)
#---------------------------
# x = np.array(x).flatten()
# x = pd.get_dummies(x)
# x = np.array(x)


######3. OneHotEncoer (2차원 받아들임)######
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder()
x = np.array(x)
x = x.reshape(-1,1)                   #가로세로 바꿈 (1,11)->(11,1) #즉, 2차원으로 받아들여야함!!(ravel()적용안됨)
x = ohe.fit_transform(x).toarray()
#x = ohe.fit_transform(np.array(x).reshape(-1,1)).toarray()
print(x)
print(x.shape)



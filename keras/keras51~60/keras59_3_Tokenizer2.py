#tokenizer : 띄어쓰기 단위별로 토큰화해서 자르겠다. (문장자르기->어절 단위->수치화)
#영어는 tokenizer좋음/ 한국어는 조사때문에 어려움 있음

from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 박민경이다. 멋있다. 또 또 얘기해봐'

token = Tokenizer()
token.fit_on_texts([text1, text2]) #2개 이상은 리스트 

print(token.word_index) #개수 가장 많은 것에 index 첫번째로 줌, 개수 같을 경우는 순서대로 indexing
#{'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '박민경이다':11, '멋있다': 12, '얘기해봐': 13}
print(token.word_counts) #단어 사용한 횟수 
#OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('지구용사', 1), ('박민경이다', 1), ('멋있다', 1), ('또', 2), ('얘기해봐', 1)])


#숫자로 바꾸는 작업(수치화)
x = token.texts_to_sequences([text1, text2])
print(x)       #[[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]  
print(type(x)) #<class 'list'>

###list합치기 (더하기) # append로도 가능###
x = x[0] + x[1]
print(x)
##########################################

#원핫인코딩 (숫자가 커질 수록 가치가 부여되는 것이 아니므로)#

######1. to_categorical ######
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# x = to_categorical(x)
# print(x)
# print(x.shape) #(18, 14) ->(18,13)


######2. get_dummies (1차원 받아들임)######
# import pandas as pd
# import numpy as np
# # x = pd.get_dummies(np.array(x).reshape(-1,))
# x = pd.get_dummies(np.array(x).ravel())
# print(x) 
# print(x.shape) #(18, 13)


######3. OneHotEncoer (2차원 받아들임)######
# from sklearn.preprocessing import OneHotEncoder
# import numpy as np
# ohe = OneHotEncoder()
# x = np.array(x)
# x = x.reshape(-1,1)                  
# x = ohe.fit_transform(x).toarray()
# #x = ohe.fit_transform(np.array(x).reshape(-1,1)).toarray()
# print(x)
# print(x.shape) #(18, 13)



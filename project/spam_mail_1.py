import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = './_data/project/'
save_path = '/_save/project/'

dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
#dt_kor = pd.read_csv(path)
#data concat(pd.concat) 
print(dt_eng)
print(dt_eng.shape) #(5171, 4)

email = dt_eng["text"]
email_y = dt_eng["label_num"]
print("이메일 최대길이", max(len(i) for i in email)) #이메일 최대길이 32258
print("이메일 평균길이", sum(map(len, email))/len(email)) #이메일 평균길이 1048.391993811642

email = email.apply(lambda x : x.lower())


#Text processing 
import nltk 
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

token = Tokenizer()
token.fit_on_texts(email)
print(token.word_index)
print(token.word_counts)

#Padding
#'post','pre'비교해보기 : email에서 스팸판별의 중요한 부분은 어디인가
email = pad_sequences(email, maxlen=6000, truncating='post') 
print(email.shape)



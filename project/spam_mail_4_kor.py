import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
import nltk
# nltk.download('punkt')
# from konlpy.tag import Okt
# from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras.models import Sequential, Model, load_model 
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Conv1D, Conv2D, LSTM, Reshape, Embedding
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


path = './_data/project/'
save_path = '/_save/project/'

#Data
dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
dt_kor = pd.read_csv(path + 'kor_spam_ham_dataset.csv')
print(dt_eng.columns)

dt_eng.drop('Unnamed: 0', axis=1, inplace= True)
dt_eng.columns = ['label', 'text', 'class']
dt_eng.head()
#dt_kor = pd.read_csv(path)
#data concat(pd.concat) 
print(dt_eng)
print(dt_eng.shape) #(5171, 3)
print(type(dt_eng)) #<class 'pandas.core.frame.DataFrame'>
# dt_eng = dt_eng.values 
# print(type(dt_eng)) #<class 'numpy.ndarray'>

print(dt_kor) #[99 rows x 3 columns]

#Eng_Text processing 
#Remove stopwords from the data
stopwords = set(stopwords.words('english'))
dt_eng['text'] = dt_eng['text'].apply(lambda x: ' '.join([ word for word in word_tokenize(x)
                                                          if not word in stopwords]))
# print(dt_eng.sample(10))
# print(dt_eng['text'][0])

#Kor_Text processing 
# tokenizer = Okt()
# data['text'] = data['text'].apply(lambda x: ' '.join(tokenizer.morphs(x)))
# tfidf = TfidfVectorizer(stop_words=['은', '는', '이', '가', '을', '를'])
# X = tfidf.fit_transform(data['text'])
# y = data['class']

# 먼저 train 데이터와 test 데이터 인덱스 없이 배열로 만들기
kor_x = np.array([x for x in dt_kor['text']])
print(kor_x)
kor_y = dt_kor.loc[:,'class']

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']
print(type(eng_x)) #<class 'pandas.core.series.Series'>

x_train, x_test, y_train, y_test = train_test_split (eng_x, eng_y, train_size=0.7, random_state=42)
print(x_train.shape, x_test.shape) #(3619,) (1552,)

kx_train, kx_test, ky_train, ky_test = train_test_split (kor_x, kor_y, train_size=0.7, random_state=42)


#vectorizer
# cVect = CountVectorizer()
tfVect = TfidfVectorizer()

#Tokenizer
vocab_size = 2000 
tokenizer = Tokenizer(num_words = vocab_size)  
tokenizer.fit_on_texts(kx_train) 
sequences_train = tokenizer.texts_to_sequences(kx_train) 
sequences_test = tokenizer.texts_to_sequences(kx_test)  
print(len(sequences_train), len(sequences_test)) #69 30



tfVect.fit(x_train)
train_engV = tfVect.transform(x_train).toarray()
test_engV = tfVect.transform(x_test).toarray()
print(train_engV.shape[0], test_engV.shape[0]) #3619 #1552
print(train_engV.shape[1], test_engV.shape[1]) #41290 # 41290


# 변환된 시퀀스 번호를 이용해 단어 임베딩 벡터 생성
word_index = tokenizer.word_index
max_length = 14
padding_type='post'
train_korx = pad_sequences(sequences_train, padding='post', maxlen=max_length)
test_korx = pad_sequences(sequences_test, padding=padding_type, maxlen=max_length)

print(train_korx.shape, test_korx.shape) #(69, 14) (30, 14)
print(train_korx)

train_engV = pad_sequences(train_engV, padding='post', maxlen=max_length)
test_engV = pad_sequences(test_engV, padding=padding_type, maxlen=max_length)
print(train_engV)

train_korx= train_korx.reshape(-1,max_length,1)
test_korx= test_korx.reshape(-1,max_length,1)
train_engV= train_engV.reshape(-1,max_length,1)
test_engV= test_engV.reshape(-1,max_length,1)



#model1
input1 = Input(shape=(14,1))
dense1 = LSTM(16, activation='relu', name='kor1')(input1)
dense2 = Dense(16, activation='relu', name='kor2')(dense1)
dense4 = Dense(16, activation='swish', name='kor4')(dense2)
output1 = Dense(16, name='output2')(dense4)
#model2
input2 = Input(shape=(14,1))
dense11 = LSTM(16, activation='relu', name='eng1')(input2)
dense12 = Dense(16, activation='relu', name='eng2')(dense11)
dense14 = Dense(16, activation='swish', name='eng4')(dense12)
output2 = Dense(16, name='output2')(dense14)

#2-3. 모델 합침 
merge1 = concatenate([output1, output2], name='mg1')  
merge2 = Dense(32, activation='selu', name='mg2')(merge1)
merge3 = Dense(32, activation='swish', name='mg3')(merge2)
merge4 = Dense(16, activation='relu', name='mg4')(merge3)
last_output = Dense(1,name='last')(merge3)

#2-4 모델 정의 
model = Model(inputs=[input1, input2], outputs=[last_output])

# model.summary()

#fit
model. compile(loss='loss', optimizer='adam')

model.fit(train_engV, y_train)
model.fit([train_korx, train_engV], y_train, epochs=1, batch_size=128, validation_split=0.2,
          )

#Predic, Evaluate
acc = model.evaluate([test_korx, test_engV], y_test)

# pred = model.predict([test_korx, test_engV])

print('Accuracy: ', acc)

'''
Accuracy:  0.9787371134020618
'''













# email = dt_eng["text"]
# email_y = dt_eng["label_num"]
# print("이메일 최대길이", max(len(i) for i in email)) #이메일 최대길이 32258
# print("이메일 평균길이", sum(map(len, email))/len(email)) #이메일 평균길이 1048.391993811642

# print(type(email)) #<class 'pandas.core.series.Series'> #df의 라인 한개 : series
# #np.array(email) : array 형식 변환  # pd.DataFrame(email) : df형식으로 변환 
# email = email.values
# email_y = email_y.values
# print(type(email)) #<class 'numpy.ndarray'>
# print(email)
# print(email_y) #[0 0 0 ... 0 0 1]

# email = email.apply(lambda x : x.lower())

#Text processing 
# import nltk 
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing import TextVectorization
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import CountVectorizer


# token = Tokenizer()
# token.fit_on_texts(email)
# print(token.word_index)
# print(token.word_counts)

# #Padding
# #'post','pre'비교해보기 : email에서 스팸판별의 중요한 부분은 어디인가
# email = pad_sequences(email, maxlen=6000, truncating='post') 
# print(email.shape)



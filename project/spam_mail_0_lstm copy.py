import numpy as np
import pandas as pd
import random
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


random.seed(42)

path = './_data/project/'
save_path = '/_save/project/'

#Data
dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
dt_kor = pd.read_csv(path + 'kor_spam_ham_dataset3.csv')
# print(dt_eng.columns)

dt_eng.drop('Unnamed: 0', axis=1, inplace= True)
dt_eng.columns = ['label', 'text', 'class']
dt_eng.head()
#dt_kor = pd.read_csv(path)
#data concat(pd.concat) 
# print(dt_eng)
# print(dt_eng.shape) #(5171, 3)
# print(type(dt_eng)) #<class 'pandas.core.frame.DataFrame'>
# dt_eng = dt_eng.values 
# print(type(dt_eng)) #<class 'numpy.ndarray'>

# print(dt_kor) #[99 rows x 3 columns]

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
# print(kor_x)
kor_y = dt_kor.loc[:,'class']

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']
print(type(eng_x)) #<class 'pandas.core.series.Series'>

train_engx, test_engx, train_engy, test_engy = train_test_split (eng_x, eng_y, train_size=0.7, random_state=42)
print(train_engx.shape, test_engx.shape) #(3619,) (1552,)

train_korx, test_korx, train_kory, test_kory = train_test_split (kor_x, kor_y, train_size=0.7, random_state=42)



# #Tokenizer
# vocab_size = 2000 
# tokenizer = Tokenizer(num_words = vocab_size)  
# tokenizer.fit_on_texts(train_korx) 
# sequences_train = tokenizer.texts_to_sequences(train_korx) 
# sequences_test = tokenizer.texts_to_sequences(test_korx)  
# print(len(sequences_train), len(sequences_test)) #69 30


#vectorizer
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
train_korx = vectorizer.fit_transform(train_korx).toarray()
test_korx = vectorizer.transform(test_korx).toarray()
train_engx = vectorizer.fit_transform(train_engx).toarray()
test_engx = vectorizer.transform(test_engx).toarray()
print(train_korx.shape, test_korx.shape) #=#(253, 1487) (109, 1487)
print(train_engx.shape, test_engx.shape) #=#(3619, 41290) (1552, 41290)



# 변환된 시퀀스 번호를 이용해 단어 임베딩 벡터 생성
# word_index = tokenizer.word_index
max_length = 230
padding_type='pre'
train_korx = pad_sequences(train_korx, padding='post', maxlen=max_length)
test_korx = pad_sequences(test_korx, padding=padding_type, maxlen=max_length)

print(train_korx.shape, test_korx.shape) #(90, 14) (39, 14)
# print(train_korx)

train_engx = pad_sequences(train_engx, padding='post', maxlen=max_length)
test_engx = pad_sequences(test_engx, padding=padding_type, maxlen=max_length)
# print(test_engx)

train_korx= train_korx.reshape(-1,max_length,1)
test_korx= test_korx.reshape(-1,max_length,1)
train_engx= train_engx.reshape(-1,max_length,1)
test_engx= test_engx.reshape(-1,max_length,1)


# Korean model1
input1 = Input(shape=(230,1))
dense1 = LSTM(16, activation='relu', name='kor1')(input1)
dense2 = Dense(16, activation='relu', name='kor2')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(16, activation='swish', name='kor4')(drop1)
output1 = Dense(16, name='output1')(dense3)
# English model2
input2 = Input(shape=(230,1))
dense11 = LSTM(16, activation='relu', name='eng1')(input2)
dense12 = Dense(16, activation='relu', name='eng2')(dense11)
drop11 = Dropout(0.2)(dense12)
dense13 = Dense(16, activation='swish', name='eng4')(drop11)
output2 = Dense(16, name='output2')(dense13)

# model merge
merge1 = concatenate([output1, output2], name='mg1')  
merge2 = Dense(32, activation='selu', name='mg2')(merge1)
merge3 = Dense(32, activation='swish', name='mg3')(merge2)
mdrop1 = Dropout(0.2)(merge3)
merge4 = Dense(16, activation='relu', name='mg4')(mdrop1)
last_output = Dense(1,activation='sigmoid', name='last')(merge4)

model = Model(inputs=[input1, input2], outputs=[last_output])

model.summary()

#fit
model. compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(train_engV, y_train)
model.fit([train_korx, train_engx], train_engy, epochs=1, batch_size=16, validation_split=0.2,)


#LSTM
#Predict, Evaluate
test_engx = test_engx[:test_korx.shape[0]]
train_engy = train_engy[:109]
acc = model.evaluate([test_korx, test_engx], train_engy)[1]
print('Accuracy: ', acc)



# Predict the label of a new email in Korean
new_email_korean = ['광고. 스팸 이메일입니다.']
new_email_english = ['This is spam email bro.']
new_email_korean = vectorizer.fit_transform(new_email_korean).toarray()
new_email_korean = pad_sequences(new_email_korean, padding='pre', maxlen=max_length)
new_email_english = vectorizer.fit_transform(new_email_english).toarray()
new_email_english = pad_sequences(new_email_english, padding='pre', maxlen=max_length)
# print(new_email_korean.shape) #(1, 230)
mail_pred = model.predict([new_email_korean,new_email_english])
print('Prediction:', mail_pred)

def pred(x):
    if x>0.5:
        return print("스팸")
    else: 
        return print("비스팸")
pred(mail_pred)
print(pred)

# y_pred = model.predict([test_korx, test_engx])
# f1 = f1_score(test_kory, y_pred)
# print('f1_score: ', f1)

# # Evaluate the performance of the model
# test_engx = test_engx[:test_korx.shape[0]]
# y_pred = model.predict([test_korx, test_engx]) 
# accuracy = accuracy_score(ky_test, y_pred)
# precision = precision_score(ky_test, y_pred)
# recall = recall_score(ky_test, y_pred)
# f1 = f1_score(ky_test, y_pred)

'''
#lstm 
Accuracy:  0.7333333492279053

#데이터 추가2
CountVectorizer()
Accuracy:  0.4871794879436493
Accuracy:  0.7948718070983887
'''

'''
#데이터 추가3 
CountVectorizer()
#max_len : 100
Accuracy:  0.7431192398071289
Accuracy:  0.7435897588729858
Accuracy:  0.752293586730957
#max_len : 230
Accuracy:  0.7981651425361633
=====================================
TfidfVectorizer()
#max_len : 230
Accuracy:  0.752293586730957
Accuracy:  0.8073394298553467
'''

'''
#데이터 추가3 
-CountVectorizer()
-max_len : 230
-epochs =1/ 100
Accuracy:  0.8073394298553467

-TfidfVectorizer()
-max_len : 230
-epochs =1/ 100
Accuracy:  0.8073394298553467
'''
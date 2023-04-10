import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
import nltk
# nltk.download('punkt')
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential, Model, load_model 
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.layers import Conv1D, Conv2D, LSTM, Reshape, Embedding
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


path = './_data/project/'
save_path = '/_save/project/'

dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
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

#Text processing 
#Remove stopwords from the data
stopwords = set(stopwords.words('english'))
dt_eng['text'] = dt_eng['text'].apply(lambda x: ' '.join([ word for word in word_tokenize(x)
                                                          if not word in stopwords]))
print(dt_eng.sample(10))

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']
print(type(eng_x)) #<class 'pandas.core.series.Series'>

x_train, x_test, y_train, y_test = train_test_split (eng_x, eng_y, train_size=0.7, random_state=42)
print(x_train.shape, x_test.shape) #(3619,) (1552,)

#vectorizer
# cVect = CountVectorizer()
tfVect = TfidfVectorizer()

tfVect.fit(x_train)
train_engV = tfVect.transform(x_train).toarray()
test_engV = tfVect.transform(x_test).toarray()
print(train_engV.shape[0], test_engV.shape[0]) #3619 #1552
print(train_engV.shape[1], test_engV.shape[1]) #41290 # 41290
print(train_engV.shape) #(3619, 41290)
# print("메일의 최대길이:", max(len(i) for i in train_engV)) #41290
# print("메일의 평균길이:", sum(map(len, train_engV))/ len(train_engV)) #41290.0

# train_engV = pad_sequences(train_engV, maxlen = 300, truncating='post')
# test_engV = pad_sequences(test_engV, maxlen = 300, truncating='post')
train_engV=train_engV.reshape(*train_engV.shape,1)
test_engV=test_engV.reshape(*test_engV.shape,1)

#model
# lr = LogisticRegression(verbose=1)

#파라미터 설정
vocab_size = 2000 # 제일 많이 사용하는 사이즈
embedding_dim = 200  
max_length = 14    # 위에서 그래프 확인 후 정함
padding_type='post'

model2 = Sequential([Embedding(vocab_size, embedding_dim, input_length =max_length),
        tf.keras.layers.LSTM(units = 64, return_sequences = True),
        tf.keras.layers.LSTM(units = 64, return_sequences = True),
        tf.keras.layers.LSTM(units = 64),
        Dense(7, activation='softmax')    # 결과값이 0~4 이므로 Dense(5)
    ])


model = Sequential()
# model.add(Input(shape=train_engV.shape[1:]))
model.add(Conv1D(64,kernel_size=10,strides=5,padding='same'))
model.add(Conv1D(16,kernel_size=10,strides=2))
model.add(LSTM(32))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#compile, fit
# model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

# model.fit(train_engV, y_train, epochs=1, batch_size=64, validation_split=0.2)

# #Predic, Evaluate
# pred = model.predict(test_engV)

# print('Accuracy: ', accuracy_score(y_test, pred))

# '''
# Accuracy:  0.9787371134020618
# '''













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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
import nltk
# nltk.download('punkt')
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
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


#Eng_Text processing 
#Remove stopwords from the data
stopwords = set(stopwords.words('english'))
dt_eng['text'] = dt_eng['text'].apply(lambda x: ' '.join([ word for word in word_tokenize(x)
                                                          if not word in stopwords]))
# print(dt_eng.sample(10))
# print(dt_eng['text'][0])

#Kor_Text processing 
tokenizer = Okt()
data['text'] = data['text'].apply(lambda x: ' '.join(tokenizer.morphs(x)))
tfidf = TfidfVectorizer(stop_words=['은', '는', '이', '가', '을', '를'])
X = tfidf.fit_transform(data['text'])
y = data['class']

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']
print(type(eng_x)) #<class 'pandas.core.series.Series'>

x_train, x_test, y_train, y_test = train_test_split (eng_x, eng_y, train_size=0.7, random_state=42)
print(x_train.shape, x_test.shape) #(3619,) (1552,)

#vectorizer
# cVect = CountVectorizer()
tfVect = TfidfVectorizer()
#두개 비교

tfVect.fit(x_train)
train_engV = tfVect.transform(x_train).toarray()
test_engV = tfVect.transform(x_test).toarray()
print(train_engV.shape[0], test_engV.shape[0]) #3619 #1552
print(train_engV.shape[1], test_engV.shape[1]) #41290 # 41290

#model
lr = LogisticRegression(verbose=1)
#모델 다른거쓰기 

#fit
lr.fit(train_engV, y_train)

#Predic, Evaluate
pred = lr.predict(test_engV)

print('Accuracy: ', accuracy_score(y_test, pred))

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



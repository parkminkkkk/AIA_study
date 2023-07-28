import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

random.seed(42)


path = './_data/project/'
save_path = '/_save/project/'

# Load Data
dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
dt_kor = pd.read_csv(path + 'kor_spam_ham_dataset3.csv')

# Preprocess the data
dt_eng.drop('Unnamed: 0', axis=1, inplace= True)
dt_eng.columns = ['label', 'text', 'class']
dt_eng.head()
#Eng_Text processing 
#Remove stopwords from the data
stopwords = set(stopwords.words('english'))
dt_eng['text'] = dt_eng['text'].apply(lambda x: ' '.join([ word for word in word_tokenize(x)
                                                          if not word in stopwords]))

# 먼저 train 데이터와 test 데이터 인덱스 없이 배열로 만들기
kor_x = np.array([x for x in dt_kor['text']])
# print(kor_x)
kor_y = dt_kor.loc[:,'class']

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']

y=np.array(dt_kor['class'])
_, y_indices = np.unique(y, return_inverse=True)
print(np.bincount(y_indices))
# Split the Korean data into training and testing sets
X_korean_train, X_korean_test, y_korean_train, y_korean_test = train_test_split(
    dt_kor['text'], y, test_size=0.3, random_state=42,stratify=y)

# Split the English data into training and testing sets
X_english_train, X_english_test, y_english_train, y_english_test = train_test_split(
    dt_eng['text'], dt_eng['class'], test_size=0.3, random_state=42,stratify=dt_eng['class'])

# Feature extraction
vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
X_korean_train_features = vectorizer.fit_transform(X_korean_train).toarray()
print(X_korean_train_features)
X_korean_test_features = vectorizer.transform(X_korean_test).toarray()
X_english_train_features = vectorizer.fit_transform(X_english_train).toarray()
X_english_test_features = vectorizer.transform(X_english_test).toarray()
print(X_korean_train_features.shape[0], X_korean_train_features.shape[1])  
print(X_english_train_features.shape[0], X_english_train_features.shape[1]) 

# 변환된 시퀀스 번호를 이용해 단어 임베딩 벡터 생성
max_length = 230
padding_type='pre'
train_korx = pad_sequences(X_korean_train_features, padding='pre', maxlen=max_length)
test_korx = pad_sequences(X_korean_test_features, padding=padding_type, maxlen=max_length)

print(train_korx.shape, test_korx.shape) #(90, 1000) (39, 1000)
# print(train_korx)

train_engx = pad_sequences(X_english_train_features, padding='pre', maxlen=max_length)
test_engx = pad_sequences(X_english_test_features, padding=padding_type, maxlen=max_length)
print(train_engx.shape, test_engx.shape) #(3619, 1000) (1552, 1000)

# Korean model1
korean_clf = LogisticRegression(random_state=42).fit(train_korx, y_korean_train)

# English model2
english_clf = LogisticRegression(random_state=42).fit(train_engx, y_english_train)

# classifier with the two models
E_clf = RandomForestClassifier()
# E_clf = GradientBoostingClassifier()

# Fit the voting classifier on the training data
E_clf.fit(train_korx, y_korean_train)

# korean_clf.summary() # 머신러닝 모델은  summary 제공x
#훈련된 모델의 요약을 찾고 있다면 모델 coef_의 속성/ 훈련된 모델의 요약을 찾고 있다면 모델 coef_의 속성
print("Korean Model Coefficients:")
print(korean_clf.coef_)

print("English Model Coefficients:")
print(english_clf.coef_)
#훈련된 모델에서 각 기능의 중요도
print("Feature Importances of the Random Forest Classifier:")
print(E_clf.feature_importances_)

#total params전체 매개변수
#총 매개변수 수는 입력 데이터의 기능 수에 따라 결정됩니다.
# 특히 매개변수의 수는 특성 수에 1(편향 항)을 더한 것과 같습니다./ 각 입력 데이터 세트의 기능 수가 인쇄
print("Number of features in train_korx:", train_korx.shape[1])
print("Number of features in train_engx:", train_engx.shape[1])
#그런 다음 각 로지스틱 회귀 모델에 대한 총 매개변수 수를 계산하기 위해 기능 수에 1을 더할 수 있습니다.
kor_params = train_korx.shape[1] + 1
eng_params = train_engx.shape[1] + 1

print("Total parameters in Korean logistic regression model:", kor_params)
print("Total parameters in English logistic regression model:", eng_params)

#랜덤 포레스트 및 그래디언트 부스팅 분류기의 경우 매개변수의 총 수는 트리 수 또는 추정기 수와 같이 모델에 대해 선택한 하이퍼 매개변수에 따라 달라집니다.
#이렇게 하면 모델의 트리 수와 입력 데이터의 기능 수를 기반으로 랜덤 포레스트 분류기의 총 매개변수 수가 인쇄
print("Total parameters in Random Forest Classifier:", E_clf.n_estimators * (train_korx.shape[1] + 1))
#결정트리의 개수 : n_estimators


# Test the voting classifier on the testing data
y_pred = E_clf.predict(test_korx)

#lr
# Evaluate the performance of the model
accuracy = accuracy_score(y_korean_test, y_pred)
precision = precision_score(y_korean_test, y_pred)
recall = recall_score(y_korean_test, y_pred)
f1 = f1_score(y_korean_test, y_pred)

print('Accuracy:', accuracy) 
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

'''
#data2
vectorizer = TfidfVectorizer()
Accuracy: 0.7692307692307693
Precision: 0.7692307692307693
Recall: 1.0
F1-score: 0.8695652173913044
'''

'''
#data3
vectorizer = TfidfVectorizer()
#max_len : 1000(임의값)
Accuracy: 0.9174311926605505
Precision: 0.9174311926605505
Recall: 1.0
F1-score: 0.9569377990430622

#max_len : 230 / 8862
Accuracy: 0.9174311926605505
Precision: 0.9174311926605505
Recall: 1.0
F1-score: 0.9569377990430622
======================================
CountVectorizer()
#max_len :  8862
Accuracy: 0.9357798165137615
Precision: 0.9345794392523364
Recall: 1.0
F1-score: 0.966183574879227

#max_len : 230 
Accuracy: 0.9174311926605505
Precision: 0.9174311926605505
Recall: 1.0
F1-score: 0.9569377990430622
'''

'''
#E_clf = GradientBoostingClassifier()
Accuracy: 0.926605504587156
Precision: 0.9259259259259259
Recall: 1.0
F1-score: 0.9615384615384615
'''




# # Predict the label of a new email in English
# new_email_english = ['Buy cheap Viagra now!']
# new_email_english_features = vectorizer.transform(new_email_english)
# new_email_english_pred = voting_clf.predict(new_email_english_features)
# print('Prediction:', new_email_english_pred)

# #[실습]# 긍정인지 부정인지 맞추기 
# x_predict = ['나는 성호가 정말 재미없다 너무 정말']
# token.fit_on_texts(x_predict)
# x_predict = token.texts_to_sequences(x_predict)
# x_predict = np.array(x_predict)
# x_predict = x_predict.reshape(-1,6,1)
# predict = model.predict([x_predict])
# print("긍정/부정", predict)


##############################
# #Tokenizer
# vocab_size = 2000 
# tokenizer = Tokenizer(num_words = vocab_size)  
# tokenizer.fit_on_texts(X_korean_train) 
# X_korean_train_features = tokenizer.texts_to_sequences(X_korean_train)
# print(X_korean_train_features)
# print(X_korean_train_features.__class__)
# print(X_korean_train_features[0].__class__)
# X_korean_test_features = tokenizer.texts_to_sequences(X_korean_test)  
# print(len(X_korean_train_features[0]), len(X_korean_test_features)) #69 30

# word_index = tokenizer.word_index
# max_length = 50
# padding_type='pre'
# train_korx = pad_sequences(X_korean_train_features, padding='pre', maxlen=max_length)
# test_korx = pad_sequences(X_korean_test_features, padding=padding_type, maxlen=max_length)

# print(train_korx)
# train_korx = test_korx[0]
# print(train_korx)

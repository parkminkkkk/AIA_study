import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

random.seed(42)

path = './_data/project/'
save_path = '/_save/project/'

# Load Data
dt_eng = pd.read_csv(path + 'spam_ham_dataset.csv')
dt_kor = pd.read_csv(path + 'kor_spam_ham_dataset.csv')

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
# kor_x = np.array([x for x in dt_kor['text']])
# print(kor_x)
kor_x = dt_kor.loc[:,'text']
kor_y = dt_kor.loc[:,'class']

eng_x = dt_eng.loc[:,'text'] 
eng_y = dt_eng.loc[:,'class']

# Split the Korean data into training and testing sets
X_korean_train, X_korean_test, y_korean_train, y_korean_test = train_test_split(
    kor_x, kor_y, shuffle=True, test_size=0.3, random_state=42)

# Split the English data into training and testing sets
X_english_train, X_english_test, y_english_train, y_english_test = train_test_split(
    eng_x, eng_y, test_size=0.3, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer()
# X_korean_train_features = vectorizer.fit_transform(X_korean_train)
X_english_train_features = vectorizer.fit_transform(X_english_train)
# X_korean_test_features = vectorizer.transform(X_korean_test)
X_english_test_features = vectorizer.transform(X_english_test)

#Tokenizer
vocab_size = 2000 
tokenizer = Tokenizer(num_words = vocab_size)  
tokenizer.fit_on_texts(X_korean_train) 
X_korean_train_features = tokenizer.texts_to_sequences(X_korean_train) 
X_korean_test_features = tokenizer.texts_to_sequences(X_korean_test)  
print(len(X_korean_train_features), len(X_korean_test_features)) #69 30

word_index = tokenizer.word_index
max_length = 50
padding_type='post'
X_korean_train_features = pad_sequences(X_korean_train_features, padding='post', maxlen=max_length)
X_korean_test_features = pad_sequences(X_korean_test_features, padding='post')
# train_engx = pad_sequences(X_english_train_features, padding='post', maxlen=max_length)
# test_engx = pad_sequences(X_english_test_features, padding='post')
print(X_korean_train_features)

# Train the Korean logistic regression model
print(X_korean_train_features.shape)

korean_clf = LogisticRegression(random_state=42).fit(X_korean_train_features, y_korean_train)

# Train the English logistic regression model
english_clf = LogisticRegression(random_state=42).fit(X_english_train_features, y_english_train)

# Create a voting classifier with the two models
voting_clf = VotingClassifier(
    estimators=[('korean', korean_clf), ('english', english_clf)], voting='soft')

# Fit the voting classifier on the training data
voting_clf.fit(X_korean_train_features, y_korean_train)

# Test the voting classifier on the testing data
y_pred = voting_clf.predict(X_korean_test_features)

# Evaluate the performance of the model
accuracy = accuracy_score(y_korean_test, y_pred)
precision = precision_score(y_korean_test, y_pred)
recall = recall_score(y_korean_test, y_pred)
f1 = f1_score(y_korean_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Predict the label of a new email in Korean
new_email_korean = ['스팸 이메일입니다.']
new_email_korean_features = vectorizer.transform(new_email_korean)
new_email_korean_pred = voting_clf.predict(new_email_korean_features)
print('Prediction:', new_email_korean_pred)

# Predict the label of a new email in English
new_email_english = ['Buy cheap Viagra now!']
new_email_english_features = vectorizer.transform(new_email_english)
new_email_english_pred = voting_clf.predict(new_email_english_features)
print('Prediction:', new_email_english_pred)
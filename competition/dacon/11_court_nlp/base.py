#TfidfVectorizer + LogisticRegression

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

path = 'd:/study/_data/court/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')


#Data Preprocessing
vectorizer = TfidfVectorizer()
def get_vector(vectorizer, df, train_mode):
    if train_mode:
        X_facts = vectorizer.fit_transform(df['facts'])
    else:
        X_facts = vectorizer.transform(df['facts'])
    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])
    
    X = np.concatenate([X_party1.todense(), X_party2.todense(), X_facts.todense()], axis=1)
    return X

X_train = get_vector(vectorizer, train, True)
Y_train = train["first_party_winner"]
X_test = get_vector(vectorizer, test, False)


#Define Model & Train
model = LogisticRegression(random_state=337)
model.fit(X_train, Y_train)

#Inference & Submission
submit = pd.read_csv(path + 'sample_submission.csv')
pred = model.predict(X_test)
submit['first_party_winner'] = pred
submit.to_csv('./_save/court/baseline_submit.csv', index=False)
print('Done')
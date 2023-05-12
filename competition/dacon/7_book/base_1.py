# Suprise 패키지 설치
#pip install scikit-surprise

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise import KNNWithZScore
# Importing other modules from Surprise
from surprise import Dataset
from surprise.model_selection import GridSearchCV


#1. 데이터
train = pd.read_csv('d:/study/_data/dacon_book/train.csv')
test = pd.read_csv('d:/study/_data/dacon_book/test.csv')

# Surprise 라이브러리용 Reader 및 Dataset 객체 생성
reader = Reader(rating_scale=(0, 10))
train = Dataset.load_from_df(train[['User-ID', 'Book-ID', 'Book-Rating']], reader)
train = train.build_full_trainset()

# SVD 모델 훈련

model = SVD()
model.fit(train)

submit = pd.read_csv('d:/study/_data/dacon_book/sample_submission.csv')
submit['Book-Rating'] = test.apply(lambda row: model.predict(row['User-ID'], row['Book-ID']).est, axis=1)
submit.to_csv('./1_submit.csv', index=False)


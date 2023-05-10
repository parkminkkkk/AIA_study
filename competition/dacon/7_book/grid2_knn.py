# Suprise 패키지 설치
#pip install scikit-surprise

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise import KNNWithZScore
# Importing other modules from Surprise
from surprise import Dataset
from surprise import KNNWithZScore
from surprise.model_selection import GridSearchCV, RandomizedSearchCV


train_data = pd.read_csv('d:/study/_data/dacon_book/train.csv')
test_data = pd.read_csv('d:/study/_data/dacon_book/test.csv')

reader = Reader(rating_scale=(0, 10))
train = Dataset.load_from_df(train_data[['User-ID', 'Book-ID', 'Book-Rating']], reader = reader)

# 비교할 파라미터 입력 
param_grid = {'k': [30, 50],
              'sim_options': {'name': ['pearson_baseline', 'cosine'],
                              'min_support': [1,2],
                              'user_based': [True, False]}
              }
              
gs = GridSearchCV(KNNWithZScore, param_grid, measures=['rmse'], cv=3)
gs.fit(train)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

submit = pd.read_csv('d:/study/_data/dacon_book/sample_submission.csv')
submit['Book-Rating'] = test_data.apply(lambda row: gs.predict(row['User-ID'], row['Book-ID']).est, axis=1)
submit.to_csv('./1_submit.csv', index=False)


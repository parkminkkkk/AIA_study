# Suprise 패키지 설치
#pip install scikit-surprise

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise import KNNWithZScore
# Importing other modules from Surprise
from surprise import Dataset
from surprise.model_selection import GridSearchCV


train_data = pd.read_csv('d:/study/_data/dacon_book/train.csv')
test_data = pd.read_csv('d:/study/_data/dacon_book/test.csv')

reader = Reader(rating_scale=(0, 10))
train = Dataset.load_from_df(train_data[['User-ID', 'Book-ID', 'Book-Rating']], reader)

trainset = train.build_full_trainset()

param_grid = {'n_epochs': [70, 80, 90],
              'lr_all': [0.005, 0.006, 0.007],
              'reg_all': [0.05, 0.07, 0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=4)
gs.fit(trainset)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

submit = pd.read_csv('d:/study/_data/dacon_book/sample_submission.csv')
submit['Book-Rating'] = test_data.apply(lambda row: gs.predict(row['User-ID'], row['Book-ID']).est, axis=1)
submit.to_csv('./1_submit.csv', index=False)


#k-fold (교차검증) - 데이터 전처리과정
#회귀
#boston, califonia, ddarung, kaggle_bike

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') 


#1. 데이터 
#1-1
datasets = fetch_california_housing()
x, y = fetch_california_housing(return_X_y=True)
#1-2
path_ddarung = './_data/dacon_ddarung/'
path_kaggle_bike = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
ddarung_test = pd.read_csv(path_ddarung + 'test.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle_bike + 'train.csv', index_col=0)
kaggle_test = pd.read_csv(path_kaggle_bike + 'test.csv', index_col=0)

data_list = [fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]
name_list = ['fetch_california_housing','load_diabetes', 'ddarung_train', 'kaggle_train']

model = RandomForestRegressor()

 # Loop through the datasets and models, fit the models, and print the results
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) 

for i in range(len(data_list)):
    if i<2:
        x, y = data_list[i](return_X_y=True) 
    elif i==2:
        x = ddarung_train.drop(['count'], axis=1)
        y = ddarung_train['count']
    elif i==3:
        x = kaggle_train.drop(['casual', 'registered', 'count'], axis=1)
        y = kaggle_train['count']

    print(f"\nResults for dataset {i+1}:")
    scores = cross_val_score(model, x, y, cv=kfold)
    print('acc:', scores, '\ncross_val_score 평균:', round(np.mean(scores),4))


'''
#model = RandomForestRegressor()
Results for dataset 1: fetch_california_housing
acc: [0.80595702 0.81171045 0.80127588 0.82513312 0.80599167] 
cross_val_score 평균: 0.81

Results for dataset 2: load_diabetes
acc: [0.42259671 0.52680124 0.25452042 0.49099767 0.4162555 ] 
cross_val_score 평균: 0.4222

Results for dataset 3: ddarung_train
acc: [0.78042138 0.84084084 0.78945915 0.70030072 0.74840265] 
cross_val_score 평균: 0.7719

Results for dataset 4: kaggle_train
acc: [0.29641329 0.26846043 0.30372076 0.27738705 0.31987488] 
cross_val_score 평균: 0.2932
'''
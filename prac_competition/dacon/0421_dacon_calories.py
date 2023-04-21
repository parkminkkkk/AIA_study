import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#. 데이터 
path = './_data/dacon_calories/'
path_save = './_save/dacon_calories/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape) 
#결측치 확인 
print(train_csv.columns)
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())
print(train_csv.shape, test_csv.shape) 


#데이터 분리 
x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv(['Calories_Burned'])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640874, test_size=0.2)


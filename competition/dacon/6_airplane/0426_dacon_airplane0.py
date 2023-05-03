#준지도학습
import random
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import log_loss
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Fixed Seed

#1. 데이터
path = 'd:/study/_data/dacon_airplane/'
path_save = './_save/dacon_airplane/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)  #[1000000 rows x 18 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)  #[1000000 rows x 17 columns]
sample_submission = pd.read_csv('d:/study/_data/dacon_airplane/sample_submission.csv', index_col = 0)


# print(train_csv.describe())
# print(train_csv.columns)
'''
Index(['Month', 'Day_of_Month', 'Estimated_Departure_Time',
       'Estimated_Arrival_Time', 'Cancelled', 'Diverted', 'Origin_Airport',
       'Origin_Airport_ID', 'Origin_State', 'Destination_Airport',
       'Destination_Airport_ID', 'Destination_State', 'Distance', 'Airline',
       'Carrier_Code(IATA)', 'Carrier_ID(DOT)', 'Tail_Number', 'Delay'],
      dtype='object')
'''
print(train_csv.isnull().sum())

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
# 컬럼의 누락된 값은 훈련 데이터에서 해당 컬럼의 최빈값으로 대체됩니다.
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = np.nan
    train_csv[col] = train_csv[col].fillna(mode)
    
    if col in test_csv.columns:
        test_csv[col] = test_csv[col].fillna(mode)
print('Done.')
print(train_csv)
print(test_csv)

# Quantify qualitative variables
# 정성적 변수는 LabelEncoder를 사용하여 숫자로 인코딩됩니다.
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_csv[i])
    train_csv[i] = le.transform(train_csv[i])
    
    for label in np.unique(test_csv[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test_csv[i] = le.transform(test_csv[i])
print('Done.')

# Remove unlabeled data
# 훈련 세트에서 레이블이 지정되지 않은 데이터가 제거되고 숫자 레이블 열이 추가됩니다.
# train = train_csv.dropna()

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train_csv.loc[:, 'Delay_num'] = train_csv['Delay'].apply(lambda x: to_number(x, column_number))
print('Done.')

train_x = train_csv.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train_csv['Delay_num']
test_x = test_csv.drop(columns=['ID'])


###결측치처리###
imputer = IterativeImputer(estimator=XGBRegressor())
train_csv = imputer.fit_transform(train_csv)
test_csv = imputer.fit_transform(test_csv)

train_csv = pd.DataFrame(train_csv)
test_csv = pd.DataFrame(test_csv)
train_csv.columns = ['Month', 'Day_of_Month', 'Estimated_Departure_Time','Estimated_Arrival_Time', 'Cancelled', 'Diverted', 'Origin_Airport',
       'Origin_Airport_ID', 'Origin_State', 'Destination_Airport', 'Destination_Airport_ID', 'Destination_State', 'Distance', 'Airline',
       'Carrier_Code(IATA)', 'Carrier_ID(DOT)', 'Tail_Number', 'Delay']
test_csv.columns = ['Month', 'Day_of_Month', 'Estimated_Departure_Time','Estimated_Arrival_Time', 'Cancelled', 'Diverted', 'Origin_Airport',
       'Origin_Airport_ID', 'Origin_State', 'Destination_Airport', 'Destination_Airport_ID', 'Destination_State', 'Distance', 'Airline',
       'Carrier_Code(IATA)', 'Carrier_ID(DOT)', 'Tail_Number']
print(train_csv)  
print(test_csv)
print(train_csv.isnull().sum())


# x = train_csv.drop(['Delay'], axis=1)
# y = train_csv['Delay']




# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(path_save + 'dacon_airplane' + date + '.csv')


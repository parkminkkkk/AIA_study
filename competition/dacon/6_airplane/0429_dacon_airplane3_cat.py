import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
# import lightgbm as lgbm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Fixed Seed

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('d:/study/_data/dacon_airplane/train.csv', 'train')
csv_to_parquet('d:/study/_data/dacon_airplane/test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('d:/study/_data/dacon_airplane/sample_submission.csv', index_col = 0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
# 컬럼의 누락된 값은 훈련 데이터에서 해당 컬럼의 최빈값으로 대체됩니다.
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

# Quantify qualitative variables
# 정성적 변수는 LabelEncoder를 사용하여 숫자로 인코딩됩니다.
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])
    
    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
print('Done.')

# Remove unlabeled data
# 훈련 세트에서 레이블이 지정되지 않은 데이터가 제거되고 숫자 레이블 열이 추가됩니다.
train = train.dropna()

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

# 교육 데이터는 교육 및 검증 세트로 분할되고 수치 기능은 StandardScaler를 사용하여 정규화됩니다.
# 모델은 GridSearchCV와 5겹 교차 검증을 사용하여 수행되는 하이퍼파라미터 튜닝과 함께 XGBClassifier를 사용하여 훈련됩니다.
# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=3377, stratify=train_y)

# Normalize numerical features
scaler = RobustScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=640)

# Model and hyperparameter tuning using GridSearchCV
model = CatBoostClassifier(random_state=3377)
#, task_type='GPU', devices='0')
# model = lgbm.LGBMClassifier(random_state=42, n_jobs=-1)

param = {'n_estimators' : 10000, 
         'learning_rate' : 0.3,
         'max_depth' : 7,
         'l2_leaf_reg' : 2,
         'loss_funtion' : 'Logloss', 
         'rsm' : 1,
        'num_boost_round' : 500}
# param_grid = {'n_estimators' : [100],
#                'learning_rate': [0.1, 0.01],
#                'max_depth': [2,10],
#                'l2_leaf_reg': [0,2,5],
#             #    'colsample_bylevel': ['None'],
#             #    ' num_boost_round' : [500],
#                'loss_function' : ['Logloss'], 
#                'rsm': [0.8]
#             }

# grid = GridSearchCV(model,
#                     param_grid,
#                     cv=cv,
#                     scoring='accuracy',
#                     n_jobs=-1,
#                     verbose=0)

model.set_params(early_stopping_rounds =50)
model.fit(train_x, train_y, 
          eval_set= [(train_x, train_y), (val_x, val_y)])

# best_model = grid.best_estimator_

# Model evaluation
val_y_pred = model.predict(val_x)
accuracy = accuracy_score(val_y, val_y_pred)
f1 = f1_score(val_y, val_y_pred, average='weighted')
precision = precision_score(val_y, val_y_pred, average='weighted')
recall = recall_score(val_y, val_y_pred, average='weighted')
logloss = log_loss(val_y, val_y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'logloss: {logloss}')


# 하이퍼파라미터 튜닝 결과를 바탕으로 최적의 모델을 선택하고 테스트 세트의 목표 변수를 예측하는 데 사용합니다.
# Model prediction
y_pred = model.predict_proba(test_x)
y_pred = np.round(y_pred, 4)
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv('./_save/dacon_airplane/' + date + '_catboost'+ str(round(logloss, 3)) +'.csv', index=True)

# print(best_model)
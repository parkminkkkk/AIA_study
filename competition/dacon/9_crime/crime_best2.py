import pandas as pd
import numpy as np
import random
import os
# import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(777)  # Seed 고정

path = 'd:/study_data/_data/dacon_crime/'
save_path ='d:/study_data/_save/dacon_crime/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train['날씨'] = train['강수량(mm)'] + train['강설량(mm)'] + train['적설량(cm)']
test['날씨'] = test['강수량(mm)'] + test['강설량(mm)'] + test['적설량(cm)']

# x_train = train.drop(['ID', 'TARGET'], axis = 1)
x_train = train.drop(['ID', 'TARGET'], axis = 1) # 걍수량, 적설량
y_train = train['TARGET']
x_test = test.drop('ID', axis = 1)

le = LabelEncoder()

# '요일'과 '범죄발생지' 특성에 대해 Label Encoding 진행
for feature in ['요일', '범죄발생지']:
    # Train 데이터에 대해 fit과 transform을 동시에 수행
    x_train[feature] = le.fit_transform(x_train[feature])
    # Test 데이터에 대해 transform만 수행
    x_test[feature] = le.transform(x_test[feature])

ordinal_features = ['요일', '범죄발생지']

# Create a new feature 'is_weekend'
x_train['is_weekend'] = x_train['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)
x_test['is_weekend'] = x_test['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)

# Create a new feature 'is_night'
x_train['is_night'] = x_train['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)
x_test['is_night'] = x_test['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)

# Create a new feature 'is_weekend_night'
x_train['is_weekend_night'] = x_train['is_weekend'] * x_train['is_night']
x_test['is_weekend_night'] = x_test['is_weekend'] * x_test['is_night']

# Feature Engineering: one-hot encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_train_ohe = ohe.fit_transform(x_train[ordinal_features])
x_test_ohe = ohe.transform(x_test[ordinal_features])

x_train = pd.concat([x_train, pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names(ordinal_features))], axis=1)
x_test = pd.concat([x_test, pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names(ordinal_features))], axis=1)

# Scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Handle Imbalanced Data
smote = SMOTE(random_state= 42, k_neighbors = 15)
x_train, y_train = smote.fit_resample(x_train, y_train)

# PCA, 랜덤오버샘플링
xgb_model = XGBClassifier(random_state= 42, use_label_encoder=False)

 # Define parameters for GridSearchCV
params_xgb = {'max_depth' : [6, 10],
               'learning_rate' : [0.1, 0.2],
               'n_estimators' : [6, 7],
               'min_child_weight' : [1, 3],
               'subsample' : [0.6, 0.8],    
               'colsample_bytree' : [0.6, 0.8],
               'max_bin' : [10, 20],
              'reg_lambda' : [1, 4],
               'reg_alpha' : [0.01, 0.1],
               'eval_metric' : ['logloss']}

grid_cv_xgb = GridSearchCV(xgb_model, param_grid=params_xgb, cv=2, n_jobs=-1, verbose=2)

# Fit the model
grid_cv_xgb.fit(x_train, y_train)

# Print best parameters and score for the model
print('XGBoost 최적 하이퍼파라미터: ', grid_cv_xgb.best_params_)
print('XGBoost 최고 예측 정확도: ', grid_cv_xgb.best_score_)

# Get the best model
xgb_best = grid_cv_xgb.best_estimator_

# Predict
pred = xgb_best.predict(x_test)

# 제출 파일을 읽어옵니다.
submit = pd.read_csv(path + 'sample_submission.csv')

# 예측한 값을 TARGET 컬럼에 할당합니다.
submit['TARGET'] = pred

#time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv(save_path + date + '_crime.csv', index= False)

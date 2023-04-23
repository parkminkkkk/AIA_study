import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime
from lightgbm import LGBMRegressor as lgbm

#1. 데이터 
path = './_data/dacon_calories/'
path_save = './_save/dacon_calories/'

# CALL DATA
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')

# ID COLUMN REMOVE
col = ['ID',
        'Height(Feet)',
        'Height(Remainder_Inches)',
        'Weight(lb)',
        'Weight_Status',
        'Gender',
        'Age'
]

# ED, BT, BPM, CB
# CB, ED, BT, BPM


# Weight_Status, Gender => NUMBER
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

train_df = train_df.drop('ID',  axis=1)
test_df = test_df.drop('ID', axis=1)

# PolynomialFeatures DATA 
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']

# SCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train, valid SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# model 13
mod_lgbm = lgbm(n_estimators=450, num_leaves=17, max_depth=9, reg_sqrt=True,
                class_weight='balanced', reg_alpha=.15,
                objective='root_mean_squared_error', random_state=42)
mod_lgbm.fit(X_train, y_train)

# valid PREDICT
y_pred_valid = mod_lgbm.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test PREDICT
X_test = test_df.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = mod_lgbm.predict(X_test_scaled)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# SUBMIT
sample_submission_df['Calories_Burned'] = y_pred_test
sample_submission_df.to_csv(path_save + date + 'submission_MLP_Poly.csv', index=False)
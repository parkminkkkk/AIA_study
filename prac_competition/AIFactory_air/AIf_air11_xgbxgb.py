
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Select subset of features
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and validation data
X = train_data[features]
X_train, X_val = train_test_split(X, train_size=0.9, random_state=5555)

# Normalize data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
test_data_norm = scaler.transform(test_data[features])

# Define first model architecture
model_1 = XGBRegressor(n_estimators=300,
    max_depth=100,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='reg:squarederror',
    random_state=42,
    use_label_encoder=False)

# Train first model
history = model_1.fit(X_train_norm, X_train_norm)
# Predict anomalies on test data using first model
test_preds_1 = model_1.predict(test_data_norm)

# Define second model architecture
model_2 = XGBRegressor(n_estimators=100,
    max_depth=50,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    use_label_encoder=False)
# Train second model on the outputs of the first model
model_2.fit(test_preds_1, test_preds_1)

# Predict anomalies on test data using second model
test_preds_2 = model_2.predict(test_preds_1)

# Combine the two predictions by taking the average
test_preds_avg = (test_preds_1 + test_preds_2) / 2

# Predict anomalies on test data
errors = np.mean(np.power(test_data_norm - test_preds_avg, 2), axis=1)
y_pred = np.where(errors >= np.percentile(errors, 95), 1, 0)

# Save submission file
submission['label'] = pd.DataFrame({'Prediction': y_pred})
print(submission.value_counts())
print(submission['label'].value_counts())

date = datetime.datetime.now().strftime("%m%d_%H%M%S")
submission.to_csv(save_path + 'air_xgb' + date + '.csv', index=False)
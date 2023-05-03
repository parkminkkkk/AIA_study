import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load the data
path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'

train_data = pd.read_csv(path + 'train_all.csv')
test_data = pd.read_csv(path + 'test_all.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Combine train and test data for preprocessing
all_data = pd.concat([train_data, test_data], axis=0)

# Perform data preprocessing
# Encode categorical columns
categorical_cols = ['연도', '일시', '측정소']
label_encoder = LabelEncoder()
for col in categorical_cols:
    all_data[col] = label_encoder.fit_transform(all_data[col])

# Split the data back into train and test
train_data = all_data[:train_data.shape[0]]
test_data = all_data[train_data.shape[0]:]

# Remove rows with missing values in target variable (PM2.5)
train_data = train_data.dropna(subset=['PM2.5'])

# Split the data into features and target
X_train = train_data.drop(['PM2.5'], axis=1).values.astype(float)
y_train = train_data['PM2.5'].values.astype(float)

X_test = test_data.drop(['PM2.5'], axis=1).values.astype(float)

# Scale the data
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
dval = xgb.DMatrix(X_val_split, label=y_val_split)
dtest = xgb.DMatrix(X_test_scaled)

# Set XGBoost parameters
params = {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'seed': 42}

# Train the XGBoost model with early stopping
model = xgb.train(params, dtrain, num_boost_round=1000,
                  early_stopping_rounds=5, evals=[(dval, 'validation')])

# Predict on the test set
y_pred = model.predict(dtest)

# Update the submission dataframe with the predicted values
submission = submission.reindex(range(len(y_pred)))
submission['PM2.5'] = y_pred

# Save the results
submission.to_csv(save_path + 'submit01.csv', index=False)
print(f'Results saved to {save_path}submit.csv')
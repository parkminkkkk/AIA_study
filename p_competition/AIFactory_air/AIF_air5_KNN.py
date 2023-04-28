import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load train and test data
path = './_data/AIFac_air/'
save_path = './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# air_inflow, air_end_temp, out_pressure, motor_current, motor_rpm, motor_temp, type

train_data = train_data.drop(['motor_vibe'], axis=1)
test_data = test_data.drop(['motor_vibe'], axis=1)

def type_to_HP(type):
    HP = [30, 20, 10, 50, 30, 30, 30, 30]
    return [HP[i] for i in type]

train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])

# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])
X_test = scaler.transform(test_data.iloc[:, :-1])

# Model Definition
model = KNeighborsClassifier(n_neighbors=37)

# Model Training
model.fit(X_train, train_data.iloc[:, -1])

# Model Prediction
y_pred = model.predict(X_test)
submission['label'] = [1 if label == 1 else 0 for label in y_pred]

# Save the results to a submission file
import datetime 
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + 'submit_air' + date + '_motor_vibe.csv', index=False)
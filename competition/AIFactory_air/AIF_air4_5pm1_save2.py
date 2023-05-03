import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Drop 'motor_current' column from train_data and test_data
train_data = train_data.drop(['motor_current'], axis=1)
test_data = test_data.drop(['motor_current'], axis=1)

# Add 'HP' column to train_data and test_data using 'type' column
def type_to_HP(type):
    HP = [30, 20, 10, 50, 30, 30, 30, 30]
    gen = (HP[i] for i in type)
    return list(gen)

train_data['HP'] = type_to_HP(train_data['type'])
test_data['HP'] = type_to_HP(test_data['type'])

# Combine train_data and test_data into data
data = pd.concat([train_data, test_data])


# Split data into X_train and X_val
X_train, X_val = train_test_split(data, test_size=0.9, random_state=337)
# Model Definition

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.iloc[:, :-1])
X_val = scaler.fit_transform(X_val.iloc[:, :-1])

n_neighbors = 46
contamination = 0.046111
lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                         contamination=contamination,
                         leaf_size=99,
                         algorithm='auto',
                         metric='chebyshev',
                         metric_params= None,
                         novelty=True,
                         p=300
                         )

# Model Training
lof.fit(X_train)

# Model Evaluation
y_pred_train = lof.predict(X_train)
y_pred_val = lof.predict(X_val)

# Save the model to a file
with open('./_save/AIair4_5_save_model.pkl', 'wb') as f:
    pickle.dump(lof, f)



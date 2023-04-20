import pandas as pd
import numpy as np
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
# data = pd.concat([train_data, test_data])

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data.iloc[:, :-1])

# Split data into X_train and X_val
X_train, X_val = train_test_split(data_normalized, test_size=0.9, random_state=337)
# Model Definition

# n_neighbors = 46
# contamination = 0.046111
# lof = LocalOutlierFactor(n_neighbors=n_neighbors,
#                          contamination=contamination,
#                          leaf_size=99,
#                          algorithm='auto',
#                          metric='chebyshev',
#                          metric_params= None,
#                          novelty=True,
#                          p=300
#                          )

# # Model Training
# lof.fit(X_train)

# y_pred_train_tuned = lof.predict(X_val)

# # Save the model to a file
# with open('./_save/AIair4_5_save_model.pkl', 'wb') as f:
#     pickle.dump(lof, f)

# Load the saved model
with open('./_save/AIair4_5_save_model.pkl', 'rb') as f:
    lof = pickle.load(f)

# Use the LOF model to predict outliers on the validation set
y_pred_train_tuned = lof.predict(X_val)

# Convert the predicted labels to binary format (1 if outlier, -1 if inlier)
y_pred_train_tuned = [1 if x == -1 else 0 for x in y_pred_train_tuned]

# Cut outlier scores to match the length of X_train
data_normalized = data_normalized[:7389]
y_pred_train_tuned = y_pred_train_tuned[:7389]

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(data_normalized, y_pred_train_tuned, test_size=0.9, random_state=337)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Use the model to predict outliers on the test set
y_pred_test = model.predict(data_normalized)
predicted_classes = np.argmax(y_pred_test, axis=1)

# Create the submission file
submission['label'] = predicted_classes
print(submission.value_counts())
print(submission['label'].value_counts())

#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air'+ date+ '_deep_temp.csv', index=False)


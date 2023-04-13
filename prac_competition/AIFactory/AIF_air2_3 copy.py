import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from keras import regularizers

# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])


# Select subset of features for Autoencoder model
features = ['air_inflow','air_end_temp','out_pressure','motor_current','motor_rpm','motor_temp','motor_vibe']

# Split data into train and validation sets
x_train, x_val = train_test_split(data[features], train_size=0.8, random_state=640)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
normal_data_scaled = scaler.fit_transform(x_train)
test_data_scaled = scaler.transform(test_data.drop('type', axis=1))

# Define the autoencoder model with binary cross-entropy loss
input_dim = train_data.shape[1]
encoding_dim = 4
input_layer = Input(shape=(input_dim,))
encoder1 = Dense(16, activation='selu')(input_layer)
encoder2 = Dense(32, activation='selu',activity_regularizer=regularizers.l1(0.001))(encoder1)
encoder2 = Dense(32, activation='selu',activity_regularizer=regularizers.l1(0.001))(encoder1)
encoder3 = Dense(16, activation='swish')(encoder2)
decoder = Dense(input_dim, activation='sigmoid')(encoder3)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# Train Autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50)
autoencoder.fit(x_train, x_train, epochs=500, batch_size=8, validation_data=(x_val, x_val), callbacks=[es])

# # Train the autoencoder on normal data only
# autoencoder.fit(train_data, train_data, epochs=500, batch_size=16, shuffle=True, validation_data=(test_data, test_data))

# Generate reconstruction errors for the test data
test_recon = autoencoder.predict(test_data)
test_errors = np.mean(np.square(test_data - test_recon), axis=1)

# Find the threshold value
threshold = np.percentile(test_errors, 95)

# Classify test data as normal or abnormal based on the threshold
test_labels = np.zeros(len(test_data))
test_labels[test_errors > threshold] = 1

# Predict the reconstruction error for the test data
test_data_pred = autoencoder.predict(test_data_scaled)
mse = np.mean(np.power(test_data_scaled - test_data_pred, 2), axis=1)

# Define a threshold for anomaly detection
threshold = np.percentile(mse, 95)

# Label the test data as normal or abnormal based on the threshold
# test_data['label'] = np.where(mse > threshold, 1, 0)

# Save the submission file
submission = test_data[['type', 'label']]
#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)


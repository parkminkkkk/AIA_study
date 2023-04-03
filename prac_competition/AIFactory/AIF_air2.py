import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score


# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')



# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
# 
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])


# Select subset of features for Autoencoder model
features = ['air_inflow','air_end_temp','motor_current','motor_rpm','motor_temp','motor_vibe']

# Split data into train and validation sets
x_train, x_val = train_test_split(data[features], train_size=0.8, random_state=640)

# Normalize data
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# Define Autoencoder model
input_layer = Input(shape=(len(features),))
encoder = Dense(4, activation='relu')(input_layer)
decoder = Dense(len(features), activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile Autoencoder model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train Autoencoder model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
autoencoder.fit(x_train, x_train, epochs=5000, batch_size=8, validation_data=(x_val, x_val), callbacks=[es])

# Predict anomalies in test data
test_data = scaler.transform(test_data[features])
predictions = autoencoder.predict(test_data)
mse = ((test_data - predictions) ** 2).mean(axis=1)
threshold = mse.mean() + mse.std() * 2  # Set threshold based on mean and standard deviation of MSE

# # Evaluate model performance
binary_predictions = [1 if x > threshold else 0 for x in mse]
# acc = accuracy_score(test_data['type'], binary_predictions)
# print('Accuracy:', acc)

# f1_score = f1_score(test_data['type'], binary_predictions, average='macro')
# print('F1 Score:', f1_score)

# Save predictions to submission file
submission['label'] = pd.DataFrame({'Prediction': binary_predictions})

#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)


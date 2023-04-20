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

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data.iloc[:, :-1])

# Split data into X_train and X_val
X_train, X_val = train_test_split(data_normalized, test_size=0.9, random_state=337)
# Model Definition

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

y_pred_train_tuned = lof.predict(X_val)


# # Save the model to a file
# with open('./_save/AIair4_2_save_model.pkl', 'wb') as f:
#     pickle.dump(lof, f)


# Load the saved model from the file
with open('./_save/AIair4_5_save_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# # Train the LocalOutlierFactor model
# lof.fit(X_train)

# Compute outlier scores for the training set
train_outlier_scores = -lof.negative_outlier_factor_

# Cut outlier scores to match the length of X_train
train_outlier_scores = train_outlier_scores[:len(X_train)]

# Add outlier scores to the training set
X_train_with_outliers = pd.concat([X_train, pd.Series(train_outlier_scores, index=X_train.index, name='outlier_score')], axis=1)

# Compute outlier scores for the validation set
val_outlier_scores = -lof.score_samples(X_val)

# Add outlier scores to the validation set
X_val_with_outliers = pd.concat([X_val, pd.Series(val_outlier_scores, index=X_val.index, name='outlier_score')], axis=1)

# Define the deep learning model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_with_outliers.shape[1]-1))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train_with_outliers.iloc[:, :-1], X_train_with_outliers.iloc[:, -1], epochs=10, batch_size=32, validation_data=(X_val_with_outliers.iloc[:, :-1], X_val_with_outliers.iloc[:, -1]))

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val_with_outliers.iloc[:, :-1], X_val_with_outliers.iloc[:, -1])
print(f'Validation loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%')

# Generate predictions for the test set
test_outlier_scores = -loaded_model.score_samples(data_normalized)
test_data_with_outliers = pd.concat([test_data, pd.Series(test_outlier_scores, index=test_data.index, name='outlier_score')], axis=1)
test_predictions = model.predict(test_data_with_outliers.iloc[:, :-1]).flatten()

# Create the submission file
submission['prob'] = test_predictions
#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air'+date+ '_deep_temp.csv', index=False)


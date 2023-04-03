import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler



# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
# ...

from keras.layers import Input, Dense
from keras.models import Model

# Define input shape
n_features=7
input_shape = (n_features, )

# Define encoding dimension
encoding_dim = 64

# Define input layer
input_layer = Input(shape=input_shape)

# Define encoding layer
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Define decoding layer
decoded = Dense(n_features, activation='sigmoid')(encoded)

# Define autoencoder model
autoencoder = Model(input_layer, decoded)

# Define encoder model
encoder = Model(input_layer, encoded)

# Define decoder input layer
decoder_input = Input(shape=(encoding_dim, ))

# Define decoder layer
decoder_layer = autoencoder.layers[-1]

# Define decoder model
decoder = Model(decoder_input, decoder_layer(decoder_input))

# Compile model
autoencoder.compile(optimizer='adam', loss='mse')
# Train isolation forest model on train data
model = IsolationForest(random_state=640874,
                        n_estimators=200, max_samples=1000, contamination=0.05, max_features=5)
                        # len(features))

model.fit(train_data)

# Predict anomalies in test data
predictions = model.predict(test_data)

# Evaluate model performance
acc = accuracy_score(test_data['type'], predictions)
print('Accuracy:', acc)

f1_score = f1_score(test_data['type'], predictions, average='macro')
print('F1 Score:', f1_score)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})

#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)
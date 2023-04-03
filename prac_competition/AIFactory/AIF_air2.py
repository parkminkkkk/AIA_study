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

# Select subset of features for IsolationForest model
features = ['air_inflow','air_end_temp','out_pressure','motor_current','motor_rpm','motor_temp','motor_vibe','type']


x_train, x_test, y_train, y_test = train_test_split(
    data[features], data['type'], train_size=0.8, random_state=640)

scaler = MaxAbsScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

# Train isolation forest model on train data
model = IsolationForest(random_state=640874,
                        n_estimators=200, max_samples=1000, contamination=0.05, max_features=len(features))

model.fit(x_train, y_train)

# Predict anomalies in test data
predictions = model.predict(test_data[features])

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
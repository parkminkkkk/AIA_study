import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer, accuracy_score

# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
# data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)

train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])

# Normalize data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(train_data)
x_test = scaler.transform(test_data)

# Define DBSCAN model
dbscan = DBSCAN(eps=0.017, min_samples=13)

# Fit DBSCAN model to training data
dbscan.fit(x_train)

# Predict anomalies in test data
y_pred = dbscan.fit_predict(x_test)
submission['label'] = [1 if label == -1 else 0 for label in y_pred]

# Evaluate model performance
print(submission.value_counts())
print(submission['label'].value_counts())

# Save submission file with timestamp
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air_'+date+ '_DBSCAN.csv', index=False)
import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# air_inflow,air_end_temp,out_pressure,motor_current,motor_rpm,motor_temp,motor_vibe,type

# train_data = train_data.drop(['motor_current'], axis=1)
# test_data = test_data.drop(['motor_current'], axis=1)
# temp_data = train_data.loc[:,'type']
# temp_data.rename(columns = {'type': 'HP'}, inplace = True)
# train_data = pd.concat([train_data,temp_data], axis=1)
# test_data = pd.concat([test_data,temp_data], axis=1)
data = pd.concat([train_data, test_data])

print(data.head(3))

# Preprocess data
# ...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
# train_data['HP']=type_to_HP(train_data['type'])
data['type']=type_to_HP(data['type'])

print(data.head(3))

X_train, X_test = train_test_split(
    data.iloc[:, :-1], test_size=0.75, random_state=42, shuffle=True)

# Define the model
# Feature Scaling
scaler = MinMaxScaler()
# X_train = scaler.fit_transform(train_data.iloc[:, :-1])
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(X_train.shape) #(2463, 7)
print(X_test.shape)  #(7389, 7)
print(X_train)


# Model Definition
model = LocalOutlierFactor(contamination= 0.041, n_neighbors=37, novelty=True)
# model = LocalOutlierFactor(n_neighbors=25, contamination=0.048)


# Model Training
model.fit(X_train)

# Model Prediction
y_pred = model.predict(X_test)
submission['label'] = [1 if label == -1 else 0 for label in y_pred[:7389]]
# submission['label'] = [1 if label == -1 else 0 for label in y_pred[2463:]]
print(submission.value_counts())
print(submission['label'].value_counts())

# Save the results to a submission file
#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air'+date+ '_datasplit0.041.csv', index=False)

'''
#gridsearch/ 'ball_tree',5, 0.01
0    7315
1      74
Name: label, dtype: int64

#(contamination= 0.049, n_neighbors=25)
=>0.883123354
#(contamination= 0.048, n_neighbors=25)
=>0.889184209
#(contamination= 0.048, n_neighbors=25)
=>0.9531917404 ('ball_tree', 'kd_tree', 'brute')

'''
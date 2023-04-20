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

# features = ['air_inflow', 'out_pressure', 'motor_current', 'motor_temp', 'motor_vibe']
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

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

# 
X_train, X_val = train_test_split(data, test_size= 0.9, random_state= 337)
print(X_train.shape, X_val.shape)


# Feature Scaling
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])


# Model Definition

n_neighbors = 46
contamination = 0.046111
lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                         contamination=contamination,
                         leaf_size=99,
                         algorithm='auto',
                         metric='chebyshev',
                         metric_params= None,
                         novelty=False,
                         p=300
                         )

# Model Training
lof.fit(X_train)

y_pred_train_tuned = lof.fit_predict(X_val)

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
# submission['label'] = [1 if label == -1 else 0 for label in y_pred[2463:]]
print(submission.value_counts())
print(submission['label'].value_counts())

# Save the results to a submission file
#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air'+date+ '_param.csv', index=False)

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
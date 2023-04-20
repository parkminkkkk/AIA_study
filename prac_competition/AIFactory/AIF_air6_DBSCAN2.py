import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

data = pd.concat([train_data, test_data])

print(data.head(3))

# Preprocess data
# ...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)

data['type']=type_to_HP(data['type'])

print(data.head(3))

# Prepare train and test data
X = train_data[features]
print(X.shape)
pca = PCA(n_components=3)
X = pca.fit_transform(X)
print(X.shape)


# 
X_train, X_val = train_test_split(data, test_size= 0.9, random_state= 337)
print(X_train.shape, X_val.shape)

#
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

# Feature Scaling
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])


# Model Definition
eps = 0.034
min_samples = 58
dbscan = DBSCAN(eps=eps,
                min_samples=min_samples,
                metric='chebyshev',
                algorithm='auto',
                leaf_size=100,
                p=300
                )

# Model Training
dbscan.fit(X_train)

y_pred_train_tuned = dbscan.fit_predict(X_val)

# 
test_data_dbscan = scaler.fit_transform(test_data[features])
y_pred_test_dbscan = dbscan.fit_predict(test_data_dbscan)
dbscan_predictions = [1 if x == -1 else 0 for x in y_pred_test_dbscan]

submission['label'] = pd.DataFrame({'Prediction': dbscan_predictions})

print(submission.value_counts())
print(submission['label'].value_counts())

# Save the results to a submission file
#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air'+date+ '_dbscan2.csv', index=False)
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
#
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, f1_score


# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])
# print(train_data.columns)

# 
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]
print(X.shape)
pca = PCA(n_components=3)
X = pca.fit_transform(X)
print(X.shape)

# 
X_train, X_val = train_test_split(X, test_size= 0.9, random_state= 337)
print(X_train.shape, X_val.shape)

#
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# 
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
y_pred_train_tuned = lof.fit_predict(X_val)

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
# submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
# print(submission.value_counts())

#######################################################################
# train_data['label']=np.zeros((train_data.shape[0]), np.int64)
# train_data = pd.DataFrame(train_data)
test_data['label'] = pd.DataFrame({'Prediction': lof_predictions})


x = test_data.drop(['label'], axis=1)
y = test_data['label']
print(x.shape, y.shape) #(9852, 8) (9852,)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=777, test_size=0.2,
# )

#2. 모델구성
model_1 = XGBClassifier()

model_1.fit(x, y)
test_data = test_data.drop(['label'], axis=1)
test_preds_1 = model_1.predict(test_data)

#
model_2 = XGBClassifier()
model_2.fit(x, y)
test_preds_2 = model_2.predict(test_data)

# Combine the two predictions by taking the average
test_preds_avg = (test_preds_1 + test_preds_2) / 2




# 평가
y_pred = model_2.predict(test_data)
# print("accuracy_score:", accuracy_score(test_data, y_predict))
# test_preds = model.predict(test_data)
# errors = np.mean(np.power(test_data - test_preds, 2), axis=1)
# y_pred = np.where(errors >= np.percentile(errors, 96.2), 1, 0)

submission['label'] = y_pred
print(submission.value_counts())
print(submission['label'].value_counts())

# Save the results to a submission file
import datetime 
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + 'submit_air' + date + '_xgbxgb_df.csv', index=False)
# acc = accuracy_score(test_data, y_predict)
# print('acc: ', acc)
# f1 = f1_score(y_test, y_predict)


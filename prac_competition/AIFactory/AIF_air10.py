import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import joblib
import numpy as np

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
pca = PCA(n_components=3, random_state=640)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])


n_neighbors = 42
contamination = 0.04588888
#n_neighbors데이터 포인트에 대한 LOF 점수를 계산할 때 고려할 이웃 수를 결정합니다. 값 이 높을수록 이상 n_neighbors값을 감지하는 능력이 향상될 수 있지만 정상 데이터 포인트를 이상값으로 잘못 식별할 위험도 증가합니다. 따라서 n_neighbors특정 문제 및 데이터를 기반으로 신중하게 조정해야 합니다.
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

# joblib.dump(lof, './_save/ai_factory/_model_ai_factory.joblib')

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]
###################################################################
for_sub=submission.copy()
for_test=test_data.copy()
print(f'subshape:{for_sub.shape} testshape: {for_test.shape}')
submission['label'] = lof_predictions
train_data['label'] = np.zeros(shape=train_data.shape[0],dtype=np.int64)
test_data['label'] = lof_predictions
print(test_data.shape,train_data.shape)

# print(submission.value_counts())
# print(submission['label'].value_counts())

for_train=np.concatenate((train_data.values,test_data.values),axis=0)
print(for_train.shape)


# 1. data prepare
# y값이 0인 데이터와 1인 데이터 분리
zero_data = for_train[for_train[:, -1] == 0]
one_data = for_train[for_train[:, -1] == 1]
num_zero = len(zero_data)
num_one = len(one_data)

from sklearn.utils import resample
one_data = np.repeat(one_data, num_zero//num_one, axis=0)
for_train=np.concatenate((zero_data,one_data),axis=0)
x = for_train[:,:-1]
y = for_train[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,stratify=y)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
for_test=scaler.transform(for_test)

# 2. model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout,Input,BatchNormalization
from keras import regularizers
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(128,activation='selu', kernel_regularizer=regularizers.l2(0.005), use_bias=False))
model.add(BatchNormalization())
model.add(Dense(256,activation='selu', kernel_regularizer=regularizers.l2(0.005), use_bias=False))
model.add(BatchNormalization())
model.add(Dense(256,activation='swish', kernel_regularizer=regularizers.l2(0.005), use_bias=False))
model.add(Dense(128,activation='swish', kernel_regularizer=regularizers.l2(0.005), use_bias=False))
model.add(BatchNormalization())
model.add(Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.005), use_bias=False))
model.add(BatchNormalization())
model.add(Dense(128,activation='selu', kernel_regularizer=regularizers.l2(0.005), use_bias=False))
model.add(Dense(1,activation='sigmoid'))

# 3. compile,training
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
vl = ReduceLROnPlateau(monitor='val_loss' ,
                       factor = 0.3,
                       patience = 10)


es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   verbose=True,
                   restore_best_weights=True)


model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10000,batch_size=len(x_train)//99
          ,callbacks=[vl,es]) 

# 4. predict,save
print(x_train.shape,for_test.shape)
y_pred=model.predict(for_test)
for_sub[for_sub.columns[-1]]=np.round(y_pred)

import datetime
now=datetime.datetime.now().strftime('%m월%d일%h시%M분')
print(for_sub.value_counts())
for_sub.to_csv(f'{save_path}{now}_vl_es_submission.csv',index=False)
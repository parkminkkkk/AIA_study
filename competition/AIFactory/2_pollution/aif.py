import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

# load data
train = pd.read_csv('d:/study/_data/AIFac_pollution/train_all.csv') # PM2.5 file
train_aws = pd.read_csv('d:/study/_data/AIFac_pollution/train_aws_all.csv') # AWS file
test = pd.read_csv('d:/study/_data/AIFac_pollution/test_all.csv')  # PM2.5 파일
test_aws = pd.read_csv('d:/study/_data/AIFac_pollution/test_aws_all.csv')  # AWS 파일
meta = pd.read_csv('d:/study/_data/AIFac_pollution/meta_all.csv') # meta 파일
submission = pd.read_csv('d:/study/_data/AIFac_pollution/answer_sample.csv')

path = 'D:/study/_data/AIFac_pollution/'   
save_path= './_save/AIFac_pollution/'

# get the nearest location
closest_places = {
    'Areum-dong': ['Sejong Geumnam', 'Sejong Goun', 'Sejong Yeonseo'],
    'Shinheung-dong': ['Sejong Goun', 'Sejongjeon', 'Sejong Yeonseo'],
    'No Eun-dong': ['O World', 'Sejong Geum-nam', 'Gyeryong'],
    'Munchang-dong': ['O-World', 'Secheon', 'Jang-dong'],
    'Eupnae-dong': ['O-World', 'Secheon', 'Jang-dong'],
    'Jeonglim-dong': ['O-World', 'Secheon', 'Gyeryong'],
    'Princess': ['Kim Sejong', 'Jungan', 'Princess'],
    'Nonsan': ['Gyeryong', 'Yanghwa', 'Nonsan'],
    'Daecheon 2-dong': ['Chunjangdae', 'Daecheon Port', 'Cheongyang'],
    'Dokgot-ri': ['Ando', 'Dangjin', 'Daesan'],
    'Dongmun-dong': ['Hongbuk', 'Taean', 'Dangjin'],
    'Mojong-dong': ['Asan', 'Seonggeo', 'Yesan'],
    'Shinbang-dong': ['Seonggeo', 'Sejongjeon', 'Asan'],
    'Yesan-gun': ['Yugu', 'Yesan', 'Asan'],
    'Wonmyeon Lee': ['Daesan', 'Taean', 'Ando'],
    'Hongseong-eup': ['Hongseong Jukdo', 'Hongbuk', 'Yesan'],
    'Seongseong-dong': ['Seonggeo', 'Sejongjeon', 'Asan']}

# Make the branch in aws the same name as the station in train/test.
train_aws = train_aws.rename(columns={"Branch": "Station"})
test_aws = test_aws.rename(columns={"Branch": "Measuring Station"})

# merge train and train_aws datasets by station
merged_train = pd.merge(train, train_aws, on=['year', 'date'], how='outer')


# merge with test and test_aws
merged_test = pd.merge(test, test_aws, on=['year', 'date'], how='outer')

merged_train['Station'] = merged_train['Station_x']
for k, v in closest_places.items():
    mask = (merged_train['measuring station_y'] == k) & (merged_train['measuring station_x'].isin(v))
    merged_train.loc[mask, 'measure station'] = k
merged_train.drop(['Station_y', 'Station_x'], axis=1, inplace=True)
merged_train.rename(columns={'Station_x': 'Station'}, inplace=True)

merged_test['Station'] = merged_test['Station_x']
for k, v in closest_places.items():
    mask = (merged_test['measuring station_y'] == k) & (merged_test['measuring station_x'].isin(v))
    merged_test.loc[mask, 'measurement station'] = k
merged_test.drop(['measuring station_y', 'measuring station_x'], axis=1, inplace=True)
merged_test.rename(columns={'Station_x': 'Station'}, inplace=True)

le = LabelEncoder()
merged_train['location'] = le.fit_transform(merged_train['measuring station']) Create le in a new space (location) with the #copy concept (although you can do (measuring station) right away...)
merged_test['location'] = le.transform(merged_test['measurement station']) #Since it can change depending on the location or number of data, you need to transform the test according to the fit of the train**

train_data = merged_train.drop(['measurement station'], axis=1)
test_data = merged_test.drop(['measuring station'], axis=1)

train_data['month'] = train_data['date'].str[:2] #until 2nd (0~1st)
train_data['hour'] = train_data['date'].str[6:8]
train_data = train_data.drop(['Date'], axis=1)

test_data['month'] = test_data['date'].str[:2] #until 2nd (0~1st)
test_data['hour'] = test_data['date'].str[6:8]
test_data = test_data.drop(['Date'], axis=1)

### str -> change int###----------------------------------------------------- ------------------------------


train_data['month'] = pd.to_numeric(train_data['month']).astype('int8')
train_data['hour'] = pd.to_numeric(train_data['hour']).astype('int8')
# print(train_data.info())

test_data['month'] = pd.to_numeric(test_data['month']).astype('int8')
test_data['hour'] = pd.to_numeric(test_data['hour']).astype('int8')
# print(test_data.info())

x_submit = test_data[test_data.isna().any(axis=1)]

x_submit = x_submit.drop(['PM2.5'], axis=1)
#One. data
y = train_data['PM2.5']
x = train_data.drop(['PM2.5'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=55, shuffle=True
)

parameters = {'n_estimators' : 5,
              'learning_rate' : 0.08,
              'max_depth': 3,
              'gamma': 0,
              'min_child_weight': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state' : 337,
              'n_jobs' : -1
              }


#2. model composition
model = XGBRegressor()

#3. compilation, training
model.set_params(**parameters, #I think it's similar to compile
                 eval_metric = 'mae',
                 early_stopping_rounds = 20;
                 )

start = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 1
)
end = time. time()
print("Time taken:", round(end-start, 2),"seconds")
#4. evaluation, prediction

y_predict = model. predict(x_test)

results = model.score(x_test, y_test)
print("model.score:", results)
r2 = r2_score(y_test, y_predict)
print("r2.score:", r2)
mae = mean_absolute_error(y_test, y_predict)
print("mae.score:", mae)


y_submit = model.predict(x_submit)
y_submit = np.round(y_submit, 3)
submission = pd.read_csv(save_path + 'answer_sample.csv', index_col=None, header=0, encoding='utf-8-sig')
submission['PM2.5'] = y_submit

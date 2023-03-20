from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)

# 1.3 결측치
print(train_csv.isnull().sum())

# 1.4 라벨인코딩( object 에서 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)


# 1.5 x, y 분리
x = train_csv.drop(['SalePrice','LotFrontage'], axis=1)
y = train_csv['SalePrice']
test_csv = test_csv.drop(['LotFrontage'], axis=1)

print(x.shape)

# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=640874, shuffle=True)

# 1.7 Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#reshape
print(x_train.shape, x_test.shape) #(896, 78) (225, 78)
print(test_csv.shape)  #(1459, 78)
x_train= x_train.reshape(-1,78,1,1)
x_test= x_test.reshape(-1,78,1,1)
test_csv = test_csv.reshape(-1,78,1,1)


# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

model = Sequential()
model.add(Conv2D(8,(8,1),
                 padding='valid',
                 input_shape=(78,1,1))) 
model.add(Conv2D(filters=5, kernel_size=(2,1), 
                 padding='valid',
                 activation='relu')) 
model.add(Conv2D(16, (2,1))) 
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

#'mse'->rmse로 변경
import numpy as np
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%H%M')

y_submit = model.predict(test_csv)
import numpy as np
import pandas as pd
y_submit = pd.DataFrame(y_submit)
y_submit = y_submit.fillna(y_submit.mode()[0])
y_submit = np.array(y_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['SalePrice'] = y_submit
submission.to_csv(path_save + 'kaggle_house_cnn_' + date + '.csv')

#mean 평균값 
'''
loss :  [1697318144.0, 0.0]
r2 :  0.7513690209989305
RMSE :  41198.51855655464

Epoch 01355: early stopping
loss :  [1696221184.0, 0.0]
r2 :  0.7515296925692997
RMSE :  41185.20464761186
'''
#median 중앙값 
'''
loss :  [1739785856.0, 0.0]
r2 :  0.7451481099863801
RMSE :  41710.74136126126

Epoch 01538: early stopping
loss :  [1726132736.0, 0.0]
r2 :  0.747148057019558
RMSE :  41546.75674910334
'''
#mode 최빈값
'''
loss :  [1712166784.0, 0.0]
r2 :  0.7491938988510953
RMSE :  41378.33660111021

Epoch 01595: early stopping
loss :  [1740372224.0, 0.0]
r2 :  0.7450622079351303
RMSE :  41717.77041716147
'''



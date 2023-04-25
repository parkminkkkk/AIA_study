# 결측치처리 활용해서 성능 올려보기 !! -> 최종 성능 비교 

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
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
print(train_csv.shape) #(1121, 80)

train_csv = train_csv.fillna(train_csv.median())
test_csv = test_csv.fillna(test_csv.median())

# 1.5 x, y 분리
x = train_csv.drop(['SalePrice'], axis=1)
y = train_csv['SalePrice']
# test_csv = test_csv.drop(['LotFrontage'], axis=1)

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

# 2. 모델구성

input1 = Input(shape=(79,))
dense1 = Dense(128, activation='relu')(input1) 
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(256, activation='relu')(drop1)  
drop2 = Dropout(0.5)(dense2)                                                          
dense3 = Dense(128,activation='relu')(drop2)     
drop3 = Dropout(0.3)(dense3)                                                          
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.1, callbacks=[es])

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
date = date.strftime("%m%d_%H%M")

y_submit = model.predict(test_csv)
import numpy as np
import pandas as pd
y_submit = pd.DataFrame(y_submit)
y_submit = y_submit.fillna(y_submit.mode()[0])
y_submit = np.array(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['SalePrice'] = y_submit
submission.to_csv(path_save + 'kaggle_house_' + date + '.csv')


'''
*test_csv median
loss :  [1644365824.0, 0.0]
r2 :  0.7591257125073961
RMSE :  40550.7789556213

loss :  [1469541504.0, 0.0]
r2 :  0.7847347402527826
RMSE :  38334.60153736239

'''



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score



#1. 데이터 
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[1460 rows x 80 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1459 rows x 79 columns] #SalePrice 제외


#1-1 데이터 합쳐주기 (concat)
df_train = train_csv.drop(['SalePrice'], axis=1)
df = pd.concat((df_train,test_csv))

train_csv['SalePrice'] = np.log1p(train_csv["SalePrice"])  #로그 변환을 통해 정규성
price = train_csv['SalePrice']

#결측치(null) 확인 및 처리
null = (df.isna().sum() / len(df) * 100) #백분율로 계산 확인
null = null.drop(null[null == 0].index).sort_values(ascending=False)
print(null)

# PoolQC : 수영장 품질, nan = 존재X (99%)
df['PoolQC'] = df['PoolQC'].fillna('None')
 
# MiscFeature : 기타기능, nan = 존재X (96%)
df['MiscFeature'] = df['MiscFeature'].fillna('None')
 
# Alley : 골목 접근 유형, nan = 골목 접근 금지
df['Alley'] = df['Alley'].fillna('None')
 
# Fence : 울타리 여부, nan = 울타리 없음
df['Fence'] = df['Fence'].fillna('None')
 
# FireplaceQu : 벽난로 품질, nan = 벽난로 없음
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
 
# LotFrontage : 부동산과 연결된 거리의 직선 피트, nan = 연결된 거리 없음
df['LotFrontage'] = df['LotFrontage'].fillna(0)
 
# GarageFinish : 차고 마감재 품질, nan = 차고 없음
df['GarageFinish'] = df['GarageFinish'].fillna('None')
 
# GarageYrBlt : 차고 제작연도, nan = 차고 없음
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
 
# GarageQual : 차고 품질, nan = 차고 없음
df['GarageQual'] = df['GarageQual'].fillna('None')
 
# GarageCond : 차고 상태, nan = 차고 없음
df['GarageCond'] = df['GarageCond'].fillna('None')
 
# GarageType : 차고 유형, nan = 차고 없음
df['GarageType'] = df['GarageType'].fillna('None')
 
# 지하실 관련 카테고리형 데이터, nan = 지하실 없음
# BsmtExposure, BsmtCond, BsmtQual, BsmtFinType1, BsmtFinType2
for data in ['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']:
    df[data] = df[data].fillna('None')
    
# 지하실 관련 수치형 데이터, nan = 지하실 없음
# BsmtFullBath, BsmtHalfBath, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF
for data in ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']:
    df[data] = df[data].fillna(0)
    
# MasVnrType : 석조베니어 형태, nan = 베니어 없음
df['MasVnrType'] = df['MasVnrType'].fillna('None')
 
# MasVnrArea : 석조베니어 공간, nan = 베니어 없음
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
 
# MSZoning : RL이 제일 흔한 값이므로 결측치 RL로 변경
df['MSZoning'] = df['MSZoning'].fillna('RL')
 
# Utilities : AllPub이 가장 흔한 값이므로 결측치 AllPub으로 변경
df['Utilities'] = df['Utilities'].fillna('AllPub')
 
# Functional : 홈 기능, 가장 일반적인 Typ로 변경
df["Functional"] = df["Functional"].fillna("Typ")
 
# Exterior2nd :집 외부 덮개 (소재가 2개 이상인 경우), nan = 소재 1개만 사용
df['Exterior2nd'] = df['Exterior2nd'].fillna('None')
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
 
# Electrical : 전기시스템, 'SBrkr'이 제일 흔한 값이므로 변경
df['Electrical'] = df['Electrical'].fillna('SBrkr')
 
# KitchenQual : 주방 품질, 'TA'가 가장 흔한 값이므로 변경
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
 
# GarageCars, GarageArea : 차고의 차 개수와 차고넓이, nan = 차고없음
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
 
# SaleType : 판매 유형, 가장 흔한 값인 'WD'로 변경
df['SaleType'] = df['SaleType'].fillna('WD')

# astype()을 통해 형 변환 시켜준다.
# 판매월과 판매연도가 수치형으로 되어있어 카테고리형(str)로 타입 변경
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
 
# 주거유형이 수치형으로 되어있어 카테고리형으로 타입변경
df['MSSubClass'] = df['MSSubClass'].astype(str)
# 수치형데이터와 범주형데이터 분리
obj_df = df.select_dtypes(include='object')
num_df = df.select_dtypes(exclude='object')

# 등급이 나누어지거나, 순서가 없는 경우 모델에 잘못 반영될 수 있기 때문에 등급, 여부 칼럼만 포함
 
label_obj_list = ['Street', 'Alley','ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'CentralAir',
       'KitchenQual', 'FireplaceQu','GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MoSold', 'YrSold','SaleCondition']
       
# 카테고리형 칼럼을 라벨인코딩 (수치화, 문자를 0부터 시작하는 정수형 숫자로 바꾸어줌)
from sklearn.preprocessing import LabelEncoder
 
# encoder = LabelEncoder()
 
for obj in label_obj_list:
    encoder = LabelEncoder()
    encoder.fit(list(df[obj].values))
    df[obj] = encoder.transform(list(df[obj].values))
    
df = pd.get_dummies(df)
print(df.shape) #(2919, 264)

#1-2데이터분리(train_set)
x = train_csv.drop(['SalePrice'], axis=1)
print(x)
y = train_csv['SalePrice']
print(y)
print('y의 라벨값 :', np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=640874, test_size=0.2)

# #1-3data scaling(스케일링)
# scaler = MinMaxScaler() 
# # scaler = StandardScaler() 
# # scaler = MaxAbsScaler() 
# # scaler = RobustScaler() 
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test) 
# test_csv = scaler.transform(test_csv) 



#2. 모델구성 (함수형모델) 
input1 = Input(shape=(264,)) 
dense1 = Dense(16, activation='relu')(input1) 
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(20, activation='relu')(drop1)  
drop2 = Dropout(0.2)(dense2)                                                          
dense3 = Dense(10,activation='relu')(drop2)     
drop3 = Dropout(0.1)(dense3)                                                          
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1) 


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

#시간저장
# import datetime 
# date = datetime.datetime.now()  #현재시간 데이터에 넣어줌
# date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute

# #경로명 
# filepath = './_save/MCP/keras36/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04 : 4번째자리, .4: 소수점자리 - hist에서 가져옴 

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=3, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
#                       verbose=1, save_best_only=True,  
#                       filepath="".join([filepath, 'k27_', date, '_', filename])
#                       ) 
 
model.fit(x_train, y_train, epochs=5, batch_size=10,
          validation_split=0.2, 
          callbacks=[es]) #[mcp])

#4. 평가, 예측 
from sklearn.metrics import r2_score
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)

#'mse'->rmse로 변경
import numpy as np
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

#submission.csv 만들기 
y_submit = model.predict(test_csv)
# print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit
# print(submission)


#시간저장
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute
#시간을 문자데이터로 바꿈 : 문자로 바꿔야 파일명에 넣을 수 있음 

submission.to_csv(path_save + 'submit_house' + date+ 'csv')


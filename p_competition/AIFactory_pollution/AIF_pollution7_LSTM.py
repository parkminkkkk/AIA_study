'''
**시계열데이터 -날짜 자르기(년,월,일,시,분,초)** 
-주식의 경우 : 년,월,일,시,분,초,토,일,공휴일...

#방법1. 
-(컬럼 소량(pm) 사용/x,y데이터 만들어서 사용)
-y값 명확, PM2.5데이터별로 날짜별로 자르기, x는 2일치(48개) y는 3일치(72개) : 2일치 데이터로 3일치 P.M예측
-train데이터 y값 결측치(imputer) => 이후 여기있는 데이터 사용 (48개 다음엔 72개야, 반복하면서 예측)
-(35000,48)(35000,72) 만들기 / 날짜데이터까지 잘라서 사용할 경우 (35000,4,48)(35000,72) 
-LSTM/ Conv1D모델 사용 
-다차원 xgboost돌릴 수 없음 (머신러닝모델은 3차원 안먹힘 ) => (35000,4*48)(35000,72)해서 shape만 맞춰서 돌릴 수는 있음 

#방법2.
-라벨인코더 => 지역(수치화로 변경)
-연도별로 비교 => 코로나 시점 전/후로 총 미세먼지 양의 차이가 날 수 있음(1,2,3,4)중에서 하나가 커질 수 있음.. -> linear형태
-일시 중요 => 서쪽에서 바람불때(겨울) // 남쪽에서 바람불때(여름)  : 12,1,2월에 미세먼지 더 많을 확률 높음 : 월을 데이터로 사용가능, 그러나 일은 데이터로 사용하기에 명확하지 xx
-시간데이터 => 낮>밤 // 아침<점심 
-연도, 일시, 지역, pm(y) => 일시는 월과 일로 잘라줌 - 컬럼 4개로 늘어남 / 지역데이터는 라벨인코더 
-즉, ㅇㅇㅇㅇ일때 y값이야! 를 맞추는 모델 : 35000번해서 가중치 생성 => 이후, test데이터로 확인 

#AWS : 30개 지점 있음/ 최근접이웃 - 특정 지점이 가까이 있다면 3개중 1개 선택해서 데이터 뒤에 붙이면 됨.. (성능 더 안좋아질 수도 있음..왜냐하면 가중치 가장 높은 것은 PM2.5이므로,,)
#따라서, PM2.5가 가중되는 것... 잘 찾기... 

'''

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
path = 'd:/study/_data/AIFac_pollution/'
save_path = './_save/AIFac_pollution/'

train_data = pd.read_csv(path + 'train_all.csv')
train_awsdata = pd.read_csv(path + 'train_aws_all.csv')
test_data = pd.read_csv(path + 'test_all.csv')
test_awsdata = pd.read_csv(path + 'test_aws_all.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Combine train and test data for preprocessing
all_data = pd.concat([train_data, test_data], axis=0)

# Perform data preprocessing
# Encode categorical columns
categorical_cols = ['연도', '일시', '측정소']
label_encoder = LabelEncoder()
for col in categorical_cols:
    all_data[col] = label_encoder.fit_transform(all_data[col])

# Split the data back into train and test
train_data = all_data[:train_data.shape[0]]
test_data = all_data[train_data.shape[0]:]

# Remove rows with missing values in target variable (PM2.5)
train_data = train_data.dropna(subset=['PM2.5'])

# Split the data into features and target
X_train = train_data.drop(['PM2.5'], axis=1).values.astype(float)
y_train = train_data['PM2.5'].values.astype(float)

X_test = test_data.drop(['PM2.5'], axis=1).values.astype(float)

# Scale the data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Define the model architecture
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
hidden_layer1 = Dense(8, activation='swish')(input_layer)
hidden_layer2 = Dense(4, activation='selu')(hidden_layer1)
hidden_layer3 = Dense(8, activation='selu')(hidden_layer2)
hidden_layer4 = Dense(8, activation='relu')(hidden_layer3)
output_layer = Dense(1)(hidden_layer3)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model with early stopping
es = EarlyStopping(patience=50)
model.fit(X_train_split, y_train_split,
          epochs=1000, batch_size=64,
          validation_data=(X_val_split, y_val_split), callbacks=[es])

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Update the submission dataframe with the predicted values
submission = submission.reindex(range(len(y_pred)))
submission['PM2.5'] = y_pred

# Save the results
submission.to_csv(save_path + 'submit_dense_천.csv', index=False)
print(f'Results saved to {save_path}submit.csv')
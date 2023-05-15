import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout , LSTM
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier,KNeighborsTransformer
from sklearn.metrics import accuracy_score,r2_score
import time
import warnings

warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators

def rmse(y_test,y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

path = "d:/study/_data/dacon_book/"
path_save = "d:/study/_save/dacon_book/"

#1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(871393, 9)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (159621, 8)
'''
Index(['User-ID', 'Book-ID', 'Book-Rating', 'Age', 'Location', 
'Book-Title','Book-Author', 'Year-Of-Publication', 'Publisher'],
      dtype='object')
'''
print(train_csv.info()) 


# train데이터에서 필요한 column만 추출
train_df = train_csv[['User-ID', 'Book-ID', 'Book-Rating']]
test_df = test_csv[['User-ID', 'Book-ID']]


# User-ID와 Book-ID를 Label Encoding
from sklearn.preprocessing import LabelEncoder
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

train_df['User-ID'] = user_encoder.fit_transform(train_df['User-ID'].values)
train_df['Book-ID'] = book_encoder.fit_transform(train_df['Book-ID'].values)

# 각 column별 unique value 개수 구하기
n_users, n_books = len(train_df['User-ID'].unique()), len(train_df['Book-ID'].unique())
print(f"Number of users: {n_users}, Number of books: {n_books}")

# 평점 평균 계산
train_df['Avg-Rating'] = train_df.groupby(['User-ID'])['Book-Rating'].transform('mean')
mean_rating = np.mean(train_df['Avg-Rating'].values)

# train, test 데이터셋 구성
from sklearn.model_selection import train_test_split
train, val = train_test_split(train_df, test_size=0.1, random_state=42)

X_train = [train['User-ID'].values, train['Book-ID'].values]
y_train = train['Book-Rating'].values - train['Avg-Rating'].values  # 평점 - 평균 평점
X_val = [val['User-ID'].values, val['Book-ID'].values]
y_val = val['Book-Rating'].values - val['Avg-Rating'].values

X_test = [test_df['User-ID'].values, test_df['Book-ID'].values]

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dropout, Dense, Concatenate
from keras.regularizers import l2

# 모델 하이퍼파라미터 설정
latent_factor_dim = 20  # 잠재 특징 벡터의 차원 수
dropout_rate = 0.5  # 드롭아웃 비율
reg_lambda = 0.0001  # L2 정규화 가중치
lr = 0.001  # learning rate

# input layers
user_input = Input(shape=[1], name='user_input')
book_input = Input(shape=[1], name='book_input')

# embedding layers
user_embedding = Embedding(input_dim=n_users, output_dim=latent_factor_dim, name='user_embedding',
                           embeddings_regularizer=l2(reg_lambda))(user_input)
book_embedding = Embedding(input_dim=n_books, output_dim=latent_factor_dim, name='book_embedding',
                           embeddings_regularizer=l2(reg_lambda))(book_input)

# embedding layer에 dropout 추가
user_dropout = Dropout(dropout_rate, name="user_dropout")(user_embedding)
book_dropout = Dropout(dropout_rate, name="book_dropout")(book_embedding)

# dot product
merge_vec = Concatenate()([user_dropout, book_dropout])
dot_product = Dot(axes=2, name="dot_product")([merge_vec, merge_vec])

# add Dot product with Average-Rating
avg_rating = Input(shape=[1], name='avg_rating')
add_vec = Add(name="add_vector")([dot_product, avg_rating])

# Flatten layer
flatten_vec = Flatten()(add_vec)

# Dense layer 추가
fc1 = Dense(units=128, activation='relu')(flatten_vec)
fc1_dropout = Dropout(dropout_rate)(fc1)
fc2 = Dense(units=64, activation='relu')(fc1_dropout)
fc2_dropout = Dropout(dropout_rate)(fc2)
out = Dense(1, name='output')(fc2_dropout)

# 모델 컴파일
model = Model(inputs=[user_input, book_input, avg_rating], outputs=out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 요약 및 구조 확인
model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint

# 모델 학습
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=2, mode='min')
best_model = ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", verbose=2, save_best_only=True)

history = model.fit(x=[X_train[0], X_train[1], train['Avg-Rating'].values],
                    y=y_train,
                    batch_size=256,
                    epochs=1,
                    verbose=2,
                    validation_data=([X_val[0], X_val[1], val['Avg-Rating'].values], y_val),
                    callbacks=[early_stopping, best_model])

# 예측
model.load_weights('best_model.h5')
y_pred = model.predict(x=[X_test[0], X_test[1]])
y_pred = y_pred.reshape(-1)

# 제출 형식 파일 생성
submission_df = pd.DataFrame(test_df['ID'], columns=["ID"])
submission_df["Book-Rating"] = y_pred + mean_rating  # 원래 평균 평점을 더해줍니다.

submission_df.to_csv(path_save+'submit0515.csv', index=False)



# y_predict = model.predict(x=[X_test[0], X_test[1]])
# y_predict = y_predict.reshape(-1)
# mse = mean_squared_error(y_val, y_predict)
# print("RMSE : ", np.sqrt(mse))

# #time
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# # Submission
# save_path = './_save/'
# y_sub=model.predict(test_csv)
# sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
# sample_submission_csv[sample_submission_csv.columns[-1]]= y_sub + mean_rating
# sample_submission_csv.to_csv(save_path + 'sub_' + str(round(np.sqrt(mse),4)) + '.csv', index=False, float_format='%.0f')
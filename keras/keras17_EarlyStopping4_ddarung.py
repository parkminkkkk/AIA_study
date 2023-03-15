from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #(1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #(715, 9) count제외

# print(train_csv.columns)
# print(train_csv.info())
# print(train_csv.describe())
# print(type(train_csv))
# print(train_csv.isnull().sum())

###결측치제거### 
train_csv = train_csv.dropna() 
print(train_csv.isnull().sum())
# print(train_csv.info())
print(train_csv.shape)  #(1328, 10)

###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640874, test_size=0.2
)
print(x_train.shape, x_test.shape) # (929, 9) (399, 9) * train_size=0.7, random_state=777일 때 /count제외
print(y_train.shape, y_test.shape) # (929,) (399,)     * train_size=0.7, random_state=777일 때 //count제외


#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=9))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=300, mode='min',
              verbose=1, 
              restore_best_weights=True)

hist = model.fit(x_train, y_train,
          epochs=50000, batch_size=10,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )
#print(hist.history['val_loss'])

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

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

submission.to_csv(path_save + 'submit_0308_1754_es.csv') # 파일생성


#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red')
plt.plot(hist.history['val_loss'], marker='.', c='blue')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(('로스', '발_로스'))
plt.grid()
plt.show()

'''
1. [1754_es] 73.68점(갱신*)
Epoch 00273: early stopping/ loss :  2475.7314453125/ r2스코어 : 0.5751303841838872/ RMSE :  49.75672111342364
-patience=15, random_state=34553, Dense(16'relu',8'relu', 4'relu',1), 'mse'
2. [1930_es] 67.75점(갱신*)	
Epoch 03301: early stopping/ loss :  2060.399658203125/ r2스코어 : 0.7207531870123005/ RMSE :  45.3916234398938
-patience=300, random_state=640874, Dense(16'relu',8'relu', 4'relu',1), 'mse', batch_Size=10

'''
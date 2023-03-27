from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, MaxPooling2D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data       #datasets 안의 data    
y = datasets['target']  #datasets 안에 있는 target 가져오겠다 / x,y 의미(구조) 동일
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2
)

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#reshape
print(x_train.shape) #(16512, 8)
print(x_test.shape)  #(4128, 8)
x_train= x_train.reshape(16512,8,1)
x_test= x_test.reshape(4128,8,1)


#2. 모델구성
model = Sequential()
model.add(Conv1D(10,(2),padding='same',input_shape=(8,1))) 
model.add(Conv1D(filters=5, kernel_size=(2), padding='valid', activation='relu')) 
model.add(Conv1D(16, (2))) 
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', 
              verbose=1,
              restore_best_weights=True
             )  

hist = model.fit(x_train, y_train, epochs=1000, batch_size=64,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
1. Epoch 00137: early stopping/ loss : 0.4845503866672516/ r2스코어 :  0.6356255628593935
- patience=20, train_size=0.8, random_state=123, Dense(4,6,4,2,1),activation'relu', mse, batch_size=64

2. *MinMaxScaler()
Epoch 00258: early stopping/ loss : 0.3793175220489502 /r2스코어 :  0.7147589621125761

3. dnn->cnn
loss : 0.5435107350349426
r2스코어 :  0.5912882474754939

4. LSTM
loss : 0.686671793460846
r2스코어 :  0.4836333282473636

5. Conv1D
loss : 0.5864447355270386
r2스코어 :  0.559002399540844
'''
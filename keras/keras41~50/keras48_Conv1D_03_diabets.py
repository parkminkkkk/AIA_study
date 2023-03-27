from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Input, Conv1D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_diabetes()
x = datasets.data       #datasets 안의 data    
y = datasets['target']  #datasets 안에 있는 target 가져오겠다 / x,y 의미(구조) 동일
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
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
print(x_train.shape) #(353, 10) 
print(x_test.shape)  #(89, 10) 
x_train= x_train.reshape(353,10,1)
x_test= x_test.reshape(89,10,1)


#2. 모델구성


#함수형 모델
input1 = Input(shape=(10,1))
Conv1 = Conv1D(16,2,padding='same', activation='linear')(input1)
flat1 = Flatten()(Conv1)
dense2 = Dense(16, activation='relu')(flat1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(2)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Conv1D(10,(2),padding='same',input_shape=(10,1))) 
# model.add(Conv1D(filters=5, kernel_size=(2), padding='valid', activation='relu')) 
# model.add(Conv1D(16, (2))) 
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min',
              verbose=1, 
              restore_best_weights=True)

hist = model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=5000, batch_size=100,
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

'''
1. Epoch 00232: early stopping/ loss :  2940.769287109375/ r2스코어 : 0.5332236207425274
- patience=20, train_size=0.8, random_state=123, Dense(32,16,8,2,1), mse, batch_size=100

2. - MinMaxScaler(), 
Epoch 00328: early stopping/ loss :  2763.925537109375/ r2스코어 : 0.5612932891200189

3. dnn->cnn
loss :  5672.6640625 / r2스코어 : 0.09960108322478956

4. LSTM
loss :  5855.29833984375/ r2스코어 : 0.07061224631801277

5. Conv1D_Sequential
loss :  5670.63623046875/ r2스코어 : 0.09992293862787871

6. Conv1D_함수형
loss :  2765.767578125/ r2스코어 : 0.5610009355270187

'''
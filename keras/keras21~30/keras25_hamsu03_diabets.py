from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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
scaler.fit(x_train) #x_train범위만큼 잡아라
x_train = scaler.transform(x_train) #변환
#x_train의 변환 범위에 맞춰서 하라는 뜻이므로 scaler.fit할 필요x 
x_test = scaler.transform(x_test) #x_train의 범위만큼 잡아서 변환하라 


#2. 모델구성
# model = Sequential()
# model.add(Dense(32, activation='linear', input_dim=10))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(2))
# model.add(Dense(1))

input1 = Input(shape=(10, ))
dense1 = Dense(32, activation='linear')(input1)
dense2 = Dense(16)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(2)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
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
'''
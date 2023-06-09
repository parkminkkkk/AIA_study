
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 

#1. 데이터 
datasets = fetch_covtype()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
print('y의 라벨값 :', np.unique(y)) #y의 라벨값 : [1 2 3 4 5 6 7] 
print(y)

#1)one hot encoding 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y)
print(y.shape) #(581012, 7) 

#2)데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, train_size=0.8,stratify=y)

#3)data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#reshape
print(x_train.shape, x_test.shape) #(464809, 54) (116203, 54)
x_train= x_train.reshape(-1,9*6,1)
x_test= x_test.reshape(-1,54,1)

#2. 모델구성 
model = Sequential()
model.add(Conv1D(16,(2),padding='same',input_shape=(54,1)))
model.add(MaxPooling1D()) 
model.add(Conv1D(filters=5, kernel_size=(2), padding='valid', activation='relu')) 
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#ES추가 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True, verbose=1)

hist = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측 

results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1) #print(y_test.shape)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

#그림(그래프)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'])
plt.title('binary_crossentropy')
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()


'''
4. *RobustScaler/ Dense(256,128,256,512,7) 
Epoch 00212: early stopping
results: [0.1246686801314354, 0.9569029808044434]
acc: 0.9569030059464902
5. *함수형모델 Model
Epoch 00186: early stopping
results: [0.1247873455286026, 0.9554572701454163]
acc: 0.9554572601395833

*cnn
results: [0.6166614294052124, 0.7317194938659668]
acc: 0.7317194908909409

*LSTM
results: [0.6328988671302795, 0.729809045791626]
acc: 0.729809041074671

*Conv1D
results: [0.6512383818626404, 0.7163068056106567]
acc: 0.7163068079137371
'''
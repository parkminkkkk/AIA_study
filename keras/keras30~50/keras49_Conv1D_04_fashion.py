from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) 
# print(x_test.shape, y_test.shape)

print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

#one-hot-coding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000, 10)


#2) 이미지 스케일링 방법
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


#2. 모델구성 
model = Sequential()
model.add(Conv2D(16, (2),padding='same', input_shape=(28,28,1))) 
model.add(Conv2D(filters=5, kernel_size=(2), padding='valid', activation='relu')) 
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(2**4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2**3, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=30, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.2, 
          callbacks=(es))

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss:', results[0]) 
print('acc:', results[1]) 


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) 
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

'''
loss: 0.41247764229774475
acc: 0.8489000201225281

*Conv1D
loss: 0.5842732787132263
acc: 0.7897999882698059

*Conv2D
loss: 0.45247143507003784
acc: 0.8546000123023987
'''
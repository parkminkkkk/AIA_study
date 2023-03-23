from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28)
# print(x_test.shape, y_test.shape) #(60000,)

print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

#one-hot-coding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000, 10)

#reshape
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

#scaler
scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델구성 
model = Sequential()
# model.add(Dense(64, input_shape=(784,)))
model.add(Dense(64, input_shape=(28*28,)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
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
loss: 0.39955711364746094
acc: 0.8547999858856201
'''
#모델구성에서 reshape

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import Reshape, Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train,return_counts=True)) 
#np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)

#reshape, scaling
x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000, 10)


#2. 모델구성 
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D()) 
model.add(Conv2D(32, (3,3)))
model.add(Conv2D(10, 3)) 
model.add(MaxPooling2D()) 
model.add(Flatten())                        #(N,250)
model.add(Reshape(target_shape=(25,10)))    #(None, 25, 10)  # Reshape: 데이터의 순서, 데이터 내용은 바뀌지 않는다. 
model.add(Conv1D(10, 3))
model.add(LSTM(784))                        #(None, 100) 
model.add(Reshape(target_shape=(28,28,1)))  #2차원-> 3차원 
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()



'''
#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=100, batch_size=516, validation_split=0.2, 
          callbacks=(es))

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)
'''


'''
*LSTM
results: [0.10342352092266083, 0.9682999849319458]
acc: 0.9683

*Conv1D
results: [0.20911462604999542, 0.9420999884605408]
acc: 0.9421

*Conv2D
Epoch 00037: early stopping
results: [0.03752630949020386, 0.9896000027656555]
acc: 0.9896
'''
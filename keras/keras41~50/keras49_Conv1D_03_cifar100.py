from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    #(10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train,return_counts=True)) # (array([ 0,  1,  2,  3,  4,  5... 97, 98, 99])
 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0

#reshape
x_train = x_train.reshape(-1,32*3,32)
x_test = x_test.reshape(-1,32*3,32)

#2. 모델구성 
model = Sequential()
model.add(Conv1D(16, (2),padding='same', input_shape=(32*3,32))) 
model.add(Conv1D(filters=5, kernel_size=(2), padding='valid', activation='relu')) 
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(2**4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2**3, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련 
import time 
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=20, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=300, batch_size=128, validation_split=0.2, 
          callbacks=[es])

end_time = time.time()

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

# print(y_train[3333]) 
print('time :', round(end_time-start_time, 2))


# import matplotlib.pyplot as plt
# plt.imshow(x_train[3333])
# plt.show()

'''
results: [2.4174182415008545, 0.39879998564720154]
acc: 0.3988

*LSTM
results: [4.594874382019043, 0.014000000432133675]
acc: 0.014

*Conv1D
results: [3.664857864379883, 0.1412000060081482]
acc: 0.1412
time : 304.31

*Conv2D
results: [2.4174182415008545, 0.39879998564720154]
acc: 0.3988
'''
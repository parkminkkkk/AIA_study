from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    #(10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train,return_counts=True)) # (array([ 0,  1,  2,  3,  4,  5... 97, 98, 99])
 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#reshape
x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

#scaler
scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. 모델구성 
model = Sequential()
model.add(Dense(64, input_shape=(32*32*3,)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.summary()


#3. 컴파일, 훈련 
import time 
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=10, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, 
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
results: [0.23174336552619934, 0.9312000274658203]
acc: 0.9312

*dnn
results: [4.605471611022949, 0.009999999776482582]
acc: 0.01
'''
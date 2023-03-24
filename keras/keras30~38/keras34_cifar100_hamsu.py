from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
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

#2. 모델구성 
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3))) 
# model.add(MaxPooling2D()) 
# model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')) 
# model.add(Conv2D(128, 3))  
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu')) 
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='softmax'))  #outputlayer=unique개수

input1 = Input(shape=(32,32,3))
conv1 = Conv2D(4, (2,2), padding='same')(input1)
mp1 = MaxPooling2D()(conv1)
conv2 = Conv2D(6, (2,2), padding='valid')(mp1)
flat1 = Flatten()(conv2)
dense1 = Dense(2, activation='relu')(flat1)
dense2 = Dense(2)(dense1)
output1 = Dense(100, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)

model.summary()


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
results: [2.4174182415008545, 0.39879998564720154]
acc: 0.3988
'''
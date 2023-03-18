# [실습]

from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0

#2. 모델구성 
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3))) 
# model.add(MaxPooling2D()) #(2,2)중 가장 큰 값 뽑아서 반의 크기(14x14)로 재구성함 / Maxpooling안에 디폴트가 (2,2)로 중첩되지 않도록 설정되어있음 
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')) 
# model.add(Conv2D(128, 3))  #kernel_size=(2,2)/ (2,2)/ (2) 동일함 
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu')) 
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax')) 

input1 = Input(shape=(32,32,3))
conv1 = Conv2D(8, (2,2), padding='same')(input1)
mp1 = MaxPooling2D()(conv1)
conv2 = Conv2D(6, (2,2), padding='valid', activation='relu')(mp1)
flat1 = Flatten()(conv2)
dense1 = Dense(2, activation='relu')(flat1)
dense2 = Dense(2)(dense1)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)

model.summary()




#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=10, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, 
          callbacks=(es))

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:', acc)

print(y_train[3333]) 

import matplotlib.pyplot as plt
plt.imshow(x_train[3333], 'Greys')
plt.show()
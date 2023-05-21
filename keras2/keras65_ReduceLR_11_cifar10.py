import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.datasets import cifar10


#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.
x_test = x_test / 255.

#2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D()) #(2,2)중 가장 큰 값 뽑아서 반의 크기(14x14)로 재구성함 / Maxpooling안에 디폴트가 (2,2)로 중첩되지 않도록 설정되어있음 
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')) 
model.add(Conv2D(12, 3))  #kernel_size=(2,2)/ (2,2)/ (2) 동일함 
model.add(MaxPooling2D())
model.add(Conv2D(filters=25, kernel_size=(3,3), padding='valid', activation='relu')) 
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(18, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) 

#3. 컴파일, 훈련 
# model.compile(loss = "mse", optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate= learning_rate)
model.compile(loss = 'mse', optimizer = optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20, mode = 'min', verbose=1,)
rlr = ReduceLROnPlateau(monitor='val_loss', patience = 10, mode ='auto', verbose=1, factor=0.5)   #es, rlr의 patience는 따로 준다

model.fit(x_train, y_train, epochs =100, batch_size=32, verbose=1, validation_split=0.2,
            callbacks = [es, rlr])

#4. 평가, 예측 
results = model.evaluate(x_test, y_test)

print("lr:", learning_rate, "loss:", results)


'''
Epoch 00021: ReduceLROnPlateau reducing learning rate to 0.02500000037252903.
1250/1250 [==============================] - 11s 9ms/step - loss: 0.1795 - val_loss: 0.1797 - lr: 0.0500
Epoch 00021: early stopping
313/313 [==============================] - 2s 5ms/step - loss: 0.1800
lr: 0.1 loss: 0.17999999225139618
'''


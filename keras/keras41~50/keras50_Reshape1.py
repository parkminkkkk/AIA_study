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
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        640
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 10)        2890
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 10)          0
_________________________________________________________________
flatten (Flatten)            (None, 250)               0
_________________________________________________________________
reshape (Reshape)            (None, 25, 10)            0
_________________________________________________________________
conv1d (Conv1D)              (None, 23, 10)            310
_________________________________________________________________
lstm (LSTM)                  (None, 784)               2493120
_________________________________________________________________
reshape_1 (Reshape)          (None, 28, 28, 1)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 32)        320
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 10)                250890
=================================================================
Total params: 2,766,634
Trainable params: 2,766,634
Non-trainable params: 0
_________________________________________________________________
'''
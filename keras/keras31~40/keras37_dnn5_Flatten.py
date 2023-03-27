from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#[실습] : cnn성능보다 좋게 만들기
#1. 데이터 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train,return_counts=True)) 
#np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# #reshape
# x_train = x_train.reshape(60000,28*28)
# x_test = x_test.reshape(10000,28*28)

# #scaler
# scaler = MinMaxScaler() 
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)

# print(np.unique(y_train,return_counts=True)) 

#2. 모델구성 
model = Sequential()
model.add(Dense(10, input_shape =(28,28)))
model.add(Dense(9))
model.add(Dense(8))
model.add(Flatten())
model.add(Dense(7))
model.add(Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential"          #(None, 28, 28)
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 28, 10)            290     

 dense_1 (Dense)             (None, 28, 9)             99

 dense_2 (Dense)             (None, 28, 8)             80

 dense_3 (Dense)             (None, 28, 7)             63

 dense_4 (Dense)             (None, 28, 10)            80               #3차원 받아들인 Dense는 3차원으로 나옴/ Dense는 다차원으로 받을 수 있음 
                                                                        #이후 softmax를 위해서 Flatten으로 2차원 만들어 주기만 하면 됨
=================================================================
Total params: 612
Trainable params: 612
Non-trainable params: 0
_________________________________________________________________
'''



'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 28, 10)            290

 dense_1 (Dense)             (None, 28, 9)             99

 dense_2 (Dense)             (None, 28, 8)             80

 flatten (Flatten)           (None, 224)               0             #Flatten : 3->2차원 

 dense_3 (Dense)             (None, 7)                 1575

 dense_4 (Dense)             (None, 10)                80

=================================================================
Total params: 2,124
Trainable params: 2,124
Non-trainable params: 0
_________________________________________________________________

'''
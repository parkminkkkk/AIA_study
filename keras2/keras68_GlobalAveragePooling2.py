#GlobalAveagePooling
#cifar100  -> GAP 결과 더 좋도록 

from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np
import tensorflow as tf
tf.random.set_seed(123)
tf.keras.backend.set_learning_phase(0)

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)    #(10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train,return_counts=True)) # (array([ 0,  1,  2,  3,  4,  5... 97, 98, 99])
 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0


#2. 모델구성 
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D()) #(2,2) 중 가장 큰 값 뽑아서 반의 크기(14x14)로 재구성함 
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid', activation='relu')) 
model.add(Conv2D(32, 2))  #kernel_size=(2,2)/ (2,2)/ (2) 동일함 
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(100, activation='softmax')) #np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]-> output_dim에 '10'

model.summary()

#CNN : Conv2D의 연산량이 많음/ 그러나, Flatten한 이후 연산량이 더 많음(쭉 펴서 값을 보기 위함인데, 연산량이 많음) 
###==> GlobalAveragePooling // 연산량 더 적음 ###


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=30, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

import time
start = time.time()
model.fit(x_train, y_train, epochs=60, batch_size=128, validation_split=0.2, 
          callbacks=(es))
end = time.time()
print("걸린시간:", end-start)


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print('loss:', results[0]) #loss, metrics(acc)
print('acc:', results[1]) #loss, metrics(acc)

# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
# y_test = np.argmax(y_test, axis=1)
# acc = accuracy_score(y_test, y_pred)
# print('pred_acc:', acc)

#=====================================================================================#
# Flatten 
# 걸린시간: 87.59859561920166
# loss: 3.103184700012207
# acc: 0.24690000712871552

# Global_average_pooling2d
# 걸린시간: 94.1163318157196
# loss: 3.4250266551971436
# acc: 0.1736000031232834

#### Layer층이 잘 구성된다면, 통상 GlobalAveragePooling2D이 더 좋음... ###

# Flatten 
# Epoch 00040: early stopping
# 걸린시간: 128.93126440048218
# loss: 2.8563437461853027
# acc: 0.3237000107765198

# Global_average_pooling2d
# Total params: 26,584
# 걸린시간: 229.70199298858643
# loss: 2.9705069065093994
# acc: 0.26669999957084656






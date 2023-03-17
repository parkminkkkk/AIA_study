#Max Pooling 
#kernel_size=(3,3)를 사용해서 줄일려면 layer두번 거쳐야함 / 1epochs만 해도 엄청 느림
#수치에서 큰 수치만 가져오는 것이기 때문에 연산량이 없어서 속도가 빠름 , 성능도 좋음 

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


print(np.unique(y_train,return_counts=True)) 


#one-hot-coding
print(y_train)       #[5 0 4 ... 5 6 8]
print(y_train.shape) #(60000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)       #[[0. 0. 0. ... 0. 0. 0.]..[0.0.0]]
print(y_train.shape) #(60000, 10)


#1-1스케일링
#scaler : (이미지)0~255사이 => MinMax가 가장 괜찮/ 255로 나누는 경우도 있음 

#1) 이미지 스케일링 방법
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#print(np.max(x_train), np.min(x_train)) #1.0.0.0

# x_train = x_train.reshape(60000,28*28)/255.0 #reshape, scale 같이 씀
# x_test = x_test.reshape(10000,784)/255.0


#2) 이미지 스케일링 방법
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#2. 모델구성 
# model = Sequential()
# model.add(Conv2D(64, (2,2), padding='same', input_shape=(28,28,1))) 
# model.add(MaxPooling2D()) 
# model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid', activation='relu')) 
# model.add(Conv2D(32, 2)) 
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax')) 

input1 = Input(shape=(28,28,1))
conv1 = Conv2D(8, (2,2), padding='same')(input1)
mp1 = MaxPooling2D()(conv1)
conv2 = Conv2D(6, (2,2), padding='valid')(mp1)
flat1 = Flatten()(conv2)
dense1 = Dense(2, activation='relu')(flat1)
dense2 = Dense(2)(dense1)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary()

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






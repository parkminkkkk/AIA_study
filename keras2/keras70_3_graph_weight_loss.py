# loss와 weight의 관계 그리기 // ==> 보류 

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np
import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


print(np.unique(y_train,return_counts=True)) 
#np.unique #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#one-hot-coding
print(y_train)       #[5 0 4 ... 5 6 8]
print(y_train.shape) #(60000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)       #[[0. 0. 0. ... 0. 0. 0.]..[0.0.0]]
print(y_train.shape) #(60000, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,784)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#2. 모델구성 
model = load_model('./_save/keras70_1_mnist_graph.h5')

model.summary()


#2. 모델 - 피클 불러오기
# history 객체 로드
import pickle

with open('./_save/pickle_test/keras70_1_mnist_grape.pkl', 'rb') as f:
    hist = pickle.load(f)


###################### 시각화 ###############################################################
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

# Loss와 acc 그래프
plt.plot(hist['loss'], marker='o', c='red', label='loss')
plt.plot(hist['val_loss'], marker='o', c='blue', label='val_loss')
plt.plot(hist['acc'], marker='o', c='green', label='acc')
plt.plot(hist['val_acc'], marker='o', c='purple', label='val_acc')
plt.grid()
plt.title('Loss and acc')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()

plt.show()




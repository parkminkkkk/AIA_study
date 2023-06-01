#a04_ae1파일 for문 만들기

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터 
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape) #약 10프로의 확률을 랜덤하게 넣어줌  
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape) #약 10프로의 확률을 랜덤하게 넣어줌 
# print(x_train_noised.shape, x_test_noised.shape)  #(60000,784) (10000, 784)
# print(np.max(x_train_noised), np.min(x_train_noised)) #1.4981282905693214 -0.5437005089686505
# print(np.max(x_test_noised), np.min(x_test_noised))   #1.449460149539148 -0.49511554036498756

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #clip : 최소0, 최대1로 고정
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)   #clip : 최소0, 최대1로 고정
# print(np.max(x_train_noised), np.min(x_train_noised))      #1.0 0.0
# print(np.max(x_test_noised), np.min(x_test_noised))        #1.0 0.0 


#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

#함수 
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

hidden_layer_sizes = [1, 8, 32, 64, 154, 331, 486, 713]
models = []

for size in hidden_layer_sizes:
    print("============= node {}개 시작==============================".format(size))
    model = autoencoder(hidden_layer_size=size)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_noised, x_train, epochs=1, batch_size=64)
    models.append(model)
    
decoded_imgs = []

for model in models:
    decoded_img = model.predict(x_test_noised)
    decoded_imgs.append(decoded_img)

################################################################

from matplotlib import pyplot as plt 
import random

fig, axes = plt.subplots(9, 5, figsize=(20, 7))

# 이미지 다섯개를 무작위로 고른다. 
random_images = random.sample(range(x_test.shape[0]), 5)
output_names = ['x_test', 'decoded_imgs1', 'decoded_imgs8', 'decoded_imgs32', 'decoded_imgs64',
                'decoded_imgs154', 'decoded_imgs331', 'decoded_imgs486', 'decoded_imgs713']

# 원본(입력) 이미지를 맨 위에 그린다. 
for col_num, ax in enumerate(axes[0]):
    ax.imshow(x_test[random_images[col_num]].reshape(28,28), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(output_names[0])

for row_num, model in enumerate(models):
    decoded_imgs = model.predict(x_test_noised)
    for col_num, ax in enumerate(axes[row_num+1]):
        ax.imshow(decoded_imgs[random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(output_names[row_num+1])

plt.show()


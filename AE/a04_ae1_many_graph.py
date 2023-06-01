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

model_1 = autoencoder(hidden_layer_size=1)
model_8 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=154) #PCA 95% 성능 
model_331 = autoencoder(hidden_layer_size=331) #PCA 99% 성능
model_486 = autoencoder(hidden_layer_size=486) #PCA 99.9% 성능
model_713 = autoencoder(hidden_layer_size=713) #PCA 100% 성능


#3. 컴파일, 훈련
print("============= node 1개 시작==============================")
model_1.compile(optimizer='adam', loss='mse')
model_1.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 8개 시작==============================")
model_8.compile(optimizer='adam', loss='mse')
model_8.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 32개 시작==============================")
model_32.compile(optimizer='adam', loss='mse')
model_32.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 64개 시작==============================")
model_64.compile(optimizer='adam', loss='mse')
model_64.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 154개 시작==============================")
model_154.compile(optimizer='adam', loss='mse')
model_154.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 331개 시작==============================")
model_331.compile(optimizer='adam', loss='mse')
model_331.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 486개 시작==============================")
model_486.compile(optimizer='adam', loss='mse')
model_486.fit(x_train_noised, x_train, epochs= 5, batch_size=128)

print("============= node 713개 시작==============================")
model_713.compile(optimizer='adam', loss='mse')
model_713.fit(x_train_noised, x_train, epochs= 5, batch_size=128)


# 4. 평가, 예측 
decoded_imgs1 = model_1.predict(x_test_noised)
decoded_imgs8 = model_8.predict(x_test_noised)
decoded_imgs32 = model_32.predict(x_test_noised)
decoded_imgs64 = model_64.predict(x_test_noised)
decoded_imgs154 = model_154.predict(x_test_noised)
decoded_imgs331 = model_331.predict(x_test_noised)
decoded_imgs486 = model_486.predict(x_test_noised)
decoded_imgs713 = model_713.predict(x_test_noised)

################################################################

from matplotlib import pyplot as plt 
import random
fig, axes = plt.subplots(9, 5, figsize=(15, 15))

# 이미지 다섯개를 무작위로 고른다. 
random_images = random.sample(range(decoded_imgs1.shape[0]), 5)
outputs = [x_test, decoded_imgs1, decoded_imgs8, decoded_imgs32,decoded_imgs64,
           decoded_imgs154,decoded_imgs331, decoded_imgs486, decoded_imgs713
           ]
output_labels = ['x_test', 'decoded_imgs1', 'decoded_imgs8', 'decoded_imgs32',
                 'decoded_imgs64', 'decoded_imgs154', 'decoded_imgs331',
                 'decoded_imgs486', 'decoded_imgs713']


# 원본(입력) 이미지를 맨위에 그린다. 
for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.set_xlabel(output_labels[row_num])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

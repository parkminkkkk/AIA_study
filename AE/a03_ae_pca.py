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

# *ml33참고 : #pca의 95% 성능으로 hidden layer를 조정가능 
model = autoencoder(hidden_layer_size=154) #PCA 95% 성능 
# model = autoencoder(hidden_layer_size=331) #PCA 99% 성능
# model = autoencoder(hidden_layer_size=486) #PCA 99.9% 성능
# model = autoencoder(hidden_layer_size=713) #PCA 100% 성능


#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs= 30, batch_size=128)


# 4. 평가, 예측 
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



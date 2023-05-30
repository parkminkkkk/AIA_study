# 잡음 제거
# 기미, 주근깨 제거
# 이미지 데이터에 많이 씀, 하지만 모든 데이터에 사용 가능
# 준지도 학습
# x로 x를 훈련시킨다. (y값은 필요가 없다)
# 원본 사진 두장을 그대로 훈련이 가능하다

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 라벨 값이 필요가 없기에 
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 112.5  #255.
x_test = x_test.reshape(10000, 784).astype('float32') / 122.5    #255.

#2. 모델구성
# from keras.models import Sequential # Another Version
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
# 함수형! #

input_img = Input(shape=(784,))
# 0~1사이 값이 들어온다.
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
encoded = Dense(1024, activation='relu')(input_img)


decoded = Dense(784 ,activation='sigmoid')(encoded) # 첫번째  # x 자체
# 375/375 [==============================] - 2s 6ms/step - loss: 0.0729 - acc: 0.0130 - val_loss: 0.0738 - val_acc: 0.0134
# decoded = Dense(784 ,activation='linear')(encoded) # 첫번째  # x 자체
# decoded = Dense(784 ,activation='relu')(encoded) # 두번째    # x 자체
# decoded = Dense(784 ,activation='tanh')(encoded) # 첫번째  # x 자체

autoencoder = Model(input_img, decoded)
# 64 => 784 => 64 => 784 반복
# autoencoder.summary()

#       #
#   #   #
#   #   #
#       #
# 특성을 추출하다가 경계선이 흐릿해지는 현상이 발생할 수 있다.

#3. 컴파일, 훈련 
# autoencoder.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])
# # acc지표는 이곳에서 의미 없다.
# # sigmoid와 지표 loss를 사용할 수 있다.
# # autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

autoencoder.compile(optimizer = 'adam', loss = 'mse')
# autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')


autoencoder.fit(x_train, x_train, epochs= 30, batch_size = 128, validation_split=0.2) # x로 x를 훈련

# AutoEncoder에서 
# 초창기 GAN에서는 loss를 믿지 말라고 했다.

# 4. 평가, 예측 
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



#인코더 4개의 히든과 디코드 4개의 엑티베이션, 2개의 컴파일-로스 부분의 경우의 수 => 총 32가지의 로스 정리, 결과치 비교 
#==> 통상, sigmoid, mse많이 사용함 (sigmoid,mse조합이 잘먹힘(0~1사이이므로..))

#Dense(64)/ activation='sigmoid' / loss = 'mse'
#375/375 [==============================] - 2s 4ms/step - loss: 0.1624 - val_loss: 0.1608
#그림 잘나옴

#Dense(32)/ activation='sigmoid' / loss = 'mse'
# 375/375 [==============================] - 2s 7ms/step - loss: 0.1756 - val_loss: 0.1742

#Dense(64)/ activation='linear' / loss = 'mse'
# 375/375 [==============================] - 1s 4ms/step - loss: 0.0895 - val_loss: 0.0897
# 그림 흐릿

#Dense(64)/ activation='relu' / loss = 'mse'
# 375/375 [==============================] - 2s 6ms/step - loss: 0.0795 - val_loss: 0.0798
# 그림 흐릿/ linear보다는 선명

#Dense(64)/ activation='tanh' / loss = 'mse'
# 375/375 [==============================] - 2s 5ms/step - loss: 0.2042 - val_loss: 0.2022
# 그림 가장 흐릿.. 


#Dense(1024)/ activation='sigmoid' / loss = 'mse'
# 375/375 [==============================] - 1s 4ms/step - loss: 0.1618 - val_loss: 0.1596
# 잘나옴
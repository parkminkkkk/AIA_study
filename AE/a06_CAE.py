# a3_ae2를 카피해서 모델 직접구성 
# 인코더 : conv, maxpool, conv, maxpool
# 디코더 : conv, UpSampling2D(2,2), conv, UpSampling2D(2,2)
# *upsampling(2,2) : 2배로 늘어남  
#  UpSampling2D : interpolate(선형보간)방식으로 upsampling채워짐 

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터 
(x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.reshape(60000, 784).astype('float32') / 255.
# x_test = x_test.reshape(10000, 784).astype('float32') / 255.
x_train = x_train / 255.
x_test = x_test / 255.


x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape) #약 10프로의 확률을 랜덤하게 넣어줌  
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape) #약 10프로의 확률을 랜덤하게 넣어줌 

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #clip : 최소0, 최대1로 고정
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)   #clip : 최소0, 최대1로 고정


#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D

#함수 
def autoencoder():
    model = Sequential()
    #인코더
    model.add(Conv2D(16,(3,3), activation='relu', padding='same', input_shape=(28,28,1)))
    model.add(MaxPool2D(2,2))      #(N,14,14,16)
    model.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D())         #(N,7,7,8) 

    #디코더 
    model.add(Conv2D(8, (3,3), activation='relu', padding='same')) 
    model.add(UpSampling2D())      #(N, 14,14,8)
    model.add(Conv2D(16, (3,3), activation='relu', padding='same')) 
    model.add(UpSampling2D())         #(N, 28,28,16)
    model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same')) #(N, 28,28,1) 최종적으로 처음 shape과 동일하게 만들어줌
    #UpSampling2D : interpolate(선형보간)방식으로 upsampling채워짐 
    return model
# model.summary()

model = autoencoder()

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs= 3, batch_size=128)


# 4. 평가, 예측 
decoded_imgs = model.predict(x_test_noised)
print(decoded_imgs.shape)

################################################################

from matplotlib import pyplot as plt 
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) =\
      plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯개를 무작위로 고른다. 
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

      
# 원본(입력) 이미지를 맨위에 그린다. 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈를 넣은 이미지 
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다. 
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
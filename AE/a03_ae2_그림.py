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
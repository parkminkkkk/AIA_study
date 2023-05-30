import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터 
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape) #약 10프로의 확률을 랜덤하게 넣어줌  
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape) #약 10프로의 확률을 랜덤하게 넣어줌 
#np.random.normal : 노이즈는 평균이 0이고 표준 편차가 0.1인 정규 분포에서 생성
#np.random.uniform : 음수값 없도록 노이즈 줌

print(x_train_noised.shape, x_test_noised.shape)  #(60000,784) (10000, 784)

print(np.max(x_train_noised), np.min(x_train_noised)) #1.4981282905693214 -0.5437005089686505
print(np.max(x_test_noised), np.min(x_test_noised))   #1.449460149539148 -0.49511554036498756

# 0~1사이 값으로 변경 (정규분포 0~0.1사이의 값이 더해졌으므로, 변경해주어도 큰 차이 없음)
# maximum(0, x)   minimum(x,1)  #함수 두번 써주면 0~1사이로 수렴
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #clip : 최소0, 최대1로 고정
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)   #clip : 최소0, 최대1로 고정

print(np.max(x_train_noised), np.min(x_train_noised))      #1.0 0.0
print(np.max(x_test_noised), np.min(x_test_noised))        #1.0 0.0 

#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
encoded = Dense(1024, activation='relu')(input_img)


decoded = Dense(784 ,activation='sigmoid')(encoded) # 첫번째  # x 자체
# decoded = Dense(784 ,activation='linear')(encoded) # 첫번째  # x 자체
# decoded = Dense(784 ,activation='relu')(encoded) # 두번째    # x 자체
# decoded = Dense(784 ,activation='tanh')(encoded) # 첫번째  # x 자체

autoencoder = Model(input_img, decoded)
# autoencoder.summary()


#3. 컴파일, 훈련 
autoencoder.compile(optimizer = 'adam', loss = 'mse')
# autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
autoencoder.fit(x_train_noised, x_train, epochs= 30, batch_size = 128, validation_split=0.2) # x로 x를 훈련
# autoencoder.fit(x_train_noised, x_train_noised, epochs= 30, batch_size = 128, validation_split=0.2) # 노이즈로 노이즈를 훈련 가능..


# 4. 평가, 예측 
decoded_imgs = autoencoder.predict(x_test_noised)

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

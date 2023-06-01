# [실습] keras56_4 남자여자 noise넣기
# predict : 기미 주근깨 제거 
# 5개 사진 출력 / 원본, 노이즈, 아웃풋 
# conv autoencoder 사용하기 

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 넘파이까지 저장 
path = 'd:/study/_data/men_women/'
save_path = 'd:/study/_save/men_women/'


#1. 데이터 
# #이미지 전처리 (수치화만)
# datagen = ImageDataGenerator(rescale=1./255) 

# xy = datagen.flow_from_directory(
#     'd:/study_data/_data/cat_dog/PetImages/',
#     target_size=(100,100),
#     batch_size=24998,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True)

# x = xy[0][0]
# y = xy[0][1]


x_train = np.load(save_path + 'keras56_7_x_train.npy')
x_test = np.load(save_path + 'keras56_7_x_test.npy')
# y_train = np.load(save_path + 'keras56_7_y_train.npy')
# y_test = np.load(save_path + 'keras56_7_y_test.npy')

x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape) #약 10프로의 확률을 랜덤하게 넣어줌  
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape) #약 10프로의 확률을 랜덤하게 넣어줌 

print(x_train_noised.shape, x_test_noised.shape) #(1003, 100, 100, 3) (2, 100, 100, 3)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #clip : 최소0, 최대1로 고정
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)   #clip : 최소0, 최대1로 고정

print(np.max(x_train_noised), np.min(x_train_noised))      #0.6556789953418057 0.0
print(np.max(x_test_noised), np.min(x_test_noised))        #0.4339888522954737 0.0


#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D

#함수 
def autoencoder():
    model = Sequential()
    #인코더
    model.add(Conv2D(16,(3,3), activation='relu', padding='same', input_shape=(150,150,3)))
    model.add(MaxPool2D())      #(N,14,14,16)
    model.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size =(3,3)))         #(N,7,7,8) 

    #디코더 
    model.add(Conv2D(8, (3,3), activation='relu', padding='same')) 
    model.add(UpSampling2D(size = (3,3)))      #(N, 14,14,8)
    model.add(Conv2D(16, (3,3), activation='relu', padding='same')) 
    model.add(UpSampling2D())         #(N, 28,28,16)
    model.add(Conv2D(3, (3,3), activation='sigmoid', padding='same')) #(N, 100,100,3) 최종적으로 처음 shape과 동일하게 만들어줌
    #UpSampling2D : interpolate(양선형보간)방식으로 upsampling채워짐 
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
    ax.imshow(x_test[random_images[i]])
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈를 넣은 이미지 
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]])
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다. 
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]])
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()



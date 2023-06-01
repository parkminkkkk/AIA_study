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


from tensorflow.keras.preprocessing import image
path = 'D:\study\_data\pmk.jpg'

img = image.load_img(path, target_size=(150,150))
na = image.img_to_array(img)/255.0

# x_train = x_train.reshape(-1, 150,150,3)
# x_test = x_test.reshape(-1, 150,150,3)
na = na.reshape(-1, *na.shape)

print(na.shape)
print(x_train.shape, x_test.shape) #(2316, 150, 150, 3) (993, 150, 150, 3)

noise = 0.3
x_train_noised = x_train + np.random.normal(0, noise, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, noise, size=x_test.shape)
na_noised = na + np.random.normal(0, noise, size=na.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)



from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,input_shape=(150, 150, 3),kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(154,(2,2),padding='same',activation='relu'))
    # model.add(MaxPooling2D())
    
    model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
    # model.add(UpSampling2D())
    
    model.add(Conv2D(hidden_layer_size,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(3,(2,2),padding='same',activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=154) # PCA의 95% 성능

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=3)
output = model.predict(x_test_noised)
na_pred = model.predict(na)

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16, ax17, ax18)) = \
    plt.subplots(3, 6, figsize=(20, 7))
    
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    
# 노이즈가 들어간 이미지를 그린다.
for i, ax in enumerate([ax7, ax8, ax9, ax10, ax11]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax13, ax14, ax15, ax16, ax17]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6]):
    ax.imshow(na.reshape(150, 150, 3))
    if i == 0:
        ax.set_ylabel("na", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax12]):
    ax.imshow(na_noised.reshape(150, 150, 3))
    if i == 0:
        ax.set_ylabel("noised_na", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax18]):
    ax.imshow(na_pred.reshape(150, 150, 3))
    if i == 0:
        ax.set_ylabel("pred_na", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
plt.tight_layout()
plt.show()
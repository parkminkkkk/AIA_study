# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 넘파이까지 저장 
path = 'd:/study_data/_data/cat_dog/PetImages/'
save_path = 'd:/study_data/_save/cat_dog/'

# np.save(save_path + 'karas56', arr = )

#1. time.time()으로 이미지 수치화하는 시간 체크 
#2. time.time()으로 넘파이로 변경하는 시간 체크할 것 
#고양이 666, 개 11702 깨진 파일

stt = time.time()

#1. 데이터 
#이미지 전처리 (수치화만)
datagen = ImageDataGenerator(rescale=1./255) 

xy = datagen.flow_from_directory(
    'd:/study/_data/cat_dog/PetImages/',
    target_size=(100,100),
    batch_size=24998,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True)

ett1 = time.time()
print('이미지 수치화 시간 :', np.round(ett1-stt, 2)) 
#이미지 수치화 시간 : 0.94

x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, test_size=0.3)



print(xy[0][0].shape)  #(20000, 300, 300, 3)
print(xy[0][1].shape)  #(20000,)


np.save(save_path + 'keras56_x_train.npy', arr=x_train)
np.save(save_path + 'keras56_x_test.npy', arr=x_test)
np.save(save_path + 'keras56_y_train.npy', arr=y_train)  
np.save(save_path + 'keras56_y_test.npy', arr=y_test)  

ett2 = time.time()

print('넘파이 변경 시간 :', np.round(ett2-stt, 2))
'''
넘파이 변경 시간 : 544.46
'''
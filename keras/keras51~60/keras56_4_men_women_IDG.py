#(150,150)
#(150,150)

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 넘파이까지 저장 
path = 'd:/study/_data/men_women/'
save_path = 'd:/study/_save/men_women/'


stt = time.time()

#1. 데이터 
#이미지 전처리 (수치화만)
datagen = ImageDataGenerator(rescale=1./255) 

xy = datagen.flow_from_directory(
    'd:/study/_data/men_women/',
    target_size=(150,150),
    batch_size=2520,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True)

print(xy)

ett1 = time.time()
print('이미지 수치화 시간 :', np.round(ett1-stt, 2)) 
#이미지 수치화 시간 : 0.6

x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, test_size=0.3)


print(xy[0][0].shape)  #(2520, 150, 150, 3)
print(xy[0][1].shape)  #(2520,)


np.save(save_path + 'men_x_train.npy', arr=x_train)
np.save(save_path + 'men_x_test.npy', arr=x_test)
np.save(save_path + 'women_y_train.npy', arr=y_train)  
np.save(save_path + 'women_y_test.npy', arr=y_test)  

ett2 = time.time()

print('넘파이 변경 시간 :', np.round(ett2-stt, 2))
'''
넘파이 변경 시간 : 198.42
'''
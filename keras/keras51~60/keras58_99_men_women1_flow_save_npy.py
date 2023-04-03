import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical


save_path = 'd:/study_data/_save/keras58/'
stt = time.time()

#1. 데이터 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255,) 

#D드라이브에서 데이터 가져오기 
xy_train = train_datagen.flow_from_directory( 
    'd:/study_data/_data/men_women/',
    target_size=(150,150),       
    batch_size= 2520,                    ###전체 데이터 쓸려면 160넣기(통배치)###
    class_mode='binary',         #y라벨 class(폴더별) #binary=수치화(0,1)  # categorical (5,)-> (5,2) onehotencoding 되어 나옴
    color_mode='rgb',
    # color_mode='rgb', #컬러 (5, 100, 100, 3)  #cf) rgba :투명도  (5, 100, 100, 4)
    shuffle=True,
)
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/men_women/',
    target_size=(150,150),       
    batch_size= 2520, 
    class_mode='binary',     
    color_mode='rgb', 
    shuffle=True,
)
print(xy_train[0][0].shape) #(160, 100, 100, 1)
print(xy_test[0][0].shape)  #(120, 100, 100, 1)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]


augment_size = 4000 #증폭사이즈  
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx.shape) 
print(np.min(randidx), np.max(randidx)) 

###변환해서 넣은것이 아니라 4만개가 중복이므로 .copy를 통해 중복 방지 (중복=과적합)###
x_augmented = x_train[randidx].copy() 
y_augmented =y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)

# #증폭 (이미지데이터-4차원으로 reshape)
# x_train = x_train.reshape(-1,32,32,3)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3) #데이터 양 모를때 명시가능
# x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)

#x_augmented 변환 방법2. '.next()사용'
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size=augment_size, shuffle=False
    ).next()[0]  #첫번째 튜플이 나옴(x_augmented[0]이 나옴) =>.next()[0]하면 x_augmented[0][0]까지 나옴
print(x_augmented)
print(x_augmented.shape)

#x_train, x_augmented합치기/ y_train, y_augmented합치기 
x_train = np.concatenate((x_train/255. ,x_augmented)) #x_train, x_augmented를 뒤에 엮겠다.
y_train = np.concatenate((y_train,y_augmented), axis=0)  #y는 scale하면 안됨!!!
x_test = x_test/255.
print(x_train.shape, y_train.shape) #(90000, 32, 32, 3) (90000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


ett1 = time.time()
print('이미지 증폭 시간 :', np.round(ett1-stt, 2)) 
#이미지 증폭 시간 : 20.63


np.save(save_path + 'keras58_99_x_train.npy', arr=x_train)
np.save(save_path + 'keras58_99_x_test.npy', arr=x_test)
np.save(save_path + 'keras58_99_y_train.npy', arr=y_train)  
np.save(save_path + 'keras58_99_y_test.npy', arr=y_test) 

ett2 = time.time()

print('넘파이 변경 시간 :', np.round(ett2-stt, 2))
# 넘파이 변경 시간 : 29.15



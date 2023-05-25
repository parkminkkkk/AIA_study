#[실습] keras81
#최종 지표 acc도출 
#기존/전이학습 성능 비교 
#무조건 전이학습이 이겨야 함 
#본인 사진 넣어서 개/고양이 구별 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score 

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 넘파이까지 저장 
path = 'd:/study/_data/cat_dog/PetImages/'
save_path = 'd:/study/_save/cat_dog/'

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


x_train = np.load(save_path + 'keras56_x_train.npy')
x_test = np.load(save_path + 'keras56_x_test.npy')
y_train = np.load(save_path + 'keras56_y_train.npy')
y_test = np.load(save_path + 'keras56_y_test.npy')


print(np.unique(y_train,return_counts=True)) 
# (array([0., 1.], dtype=float32), array([8663, 8835], dtype=int64))


x_train = x_train / 255.
x_test = x_test / 255.


#2. 모델 
vgg16 = VGG16(weights='imagenet',  #가중치는 이미지넷에서 가져다 사용
              include_top=False,    #include_top: (True, False), Classification Layer의 포함여부 // False : input, output(fc_dense) layer제거 (Classification Layer를 제거)
              input_shape=(100,100,3)
              ) 

vgg16.trainable = False  #False : vgg16의 가중치 동결 


model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

# model.trainable = True   ##vgg16만 가중치 동결(가져온 모델은 가중치 동결하고, 밑에 새로만든 dense는 가중치 형성해줌) 

# model.summary()
# print(len(model.weights))
# print(len(model.trainable_weights))

#3. 컴파일, 훈련 
# model.compile(loss = "mse", optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate= learning_rate)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['acc'])


model.fit(x_train, y_train, epochs =1, batch_size=512, verbose=1, validation_split=0.2)


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)

print("loss:", results[0])
print("acc:", results[1])


# vgg16.trainable = False 
# loss: 0.6931101679801941
# acc: 0.5114666819572449

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions   

path = 'D:\study\_data\pmk.jpg'

img = image.load_img(path, target_size= (100, 100, 3))
print(img)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x1F8B55382B0>

x = image.img_to_array(img)
print("==================== image.img_to_array(img) ====================")
print(x, '\n', x.shape)     # (224, 224, 3)
print(np.min(x), np.max(x)) # 0.0 255.0

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
# print(x.shape)      # (1, 224, 224, 3)

x = np.expand_dims(x, axis = 0)
print(x.shape)      #  (1, 224, 224, 3)

#################### -155에서 155 사이로 정규화 ###################
print("==================== preprocess_input(x) ====================")

# x = preprocess_input(x)

print(x.shape)      #  (1, 224, 224, 3)
print(np.min(x), np.max(x))      # -123.68 151.061

print("==================== model.predict(x) ====================")
x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)       #  (1, 1000)

x_pred = np.round(x_pred)

if x_pred ==0:
    print("나는 고양이")
    
else:
    print("나는 개")

'''
    ==================== model.predict(x) ====================
x_pred : [[5.8001358e-37]] 
x_pred.shape :  (1, 1)
나는 고양이
'''
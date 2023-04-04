from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical


#seed고정
np.random.seed(42)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

###증폭###
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)
print(x_train.shape)    #(60000, 28, 28)
print(x_train[0].shape) #(28, 28)
print(x_train[1].shape) #(28, 28)
print(x_train[0][0].shape) #(28,)

#6만개를 4만개 데이터로 증폭(총 10만개)
augment_size = 40000 #증폭사이즈  

# randidx = np.random.randint(60000, size = 40000) #랜덤하게 6만개에서 4만개 뽑을 것
randidx = np.random.randint(x_train.shape[0], size=augment_size) 
print(randidx)       #[46080  1860 15952 ...  8776 34167 38743]
print(randidx.shape) #(40000,) 스칼라 4만개 벡터1개 

print(np.min(randidx), np.max(randidx)) #2 59998/ 0 59999 *(랜덤)

###변환해서 넣은것이 아니라 4만개가 중복이므로 .copy를 통해 중복 방지 (중복=과적합)###
# x_augmented = x_train[randidx].copy()      #x_train에 4만개 데이터 넣은것 = x_augmented
# y_augmented =y_train[randidx].copy()
x_augmented = x_train.copy()  
y_augmented =y_train.copy()

print(x_augmented)
print(x_augmented.shape, y_augmented.shape) #(40000,28,28) (40000,)

#증폭 (이미지데이터-4차원으로 reshape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) #데이터 양 모를때 명시가능
# x_test = x_test.reshape(10000,28,28,1)과 동일 
x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)


'''
#x_augmented 변환 방법1. 
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size=augment_size, shuffle=False
) #y넣을 필요는 없지만 x,y쌍으로 되어있으므로 넣음

#x와 y가 합쳐진 iterator형태 됨 -> x_train과 합체 안됨 
print(x_augmented) #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000014508EA3B20>
print(x_augmented[0][0].shape) #(40000, 28, 28,1) #x_train과 합체
'''
#x_augmented 변환 방법2. '.next()사용'
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size=augment_size, shuffle=False
    ).next()[0]  #첫번째 튜플이 나옴(x_augmented[0]이 나옴) =>.next()[0]하면 x_augmented[0][0]까지 나옴
print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28, 1)

#문제
print(np.max(x_train), np.min(x_train))         #255.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0 : datagen에서 augmented는 이미 scaler를 했음
#x_train은 scale안되어있으므로 scale해주기 


#x_train, x_augmented합치기 (10만데이터)/ y_train, y_augmented합치기 

#ValueError: operands could not be broadcast together with shapes (60000,28,28,1) (40000,28,28,1)
# x_train = x_train + x_augmented
# print(x_train.shape) 

x_train = np.concatenate((x_train/255. ,x_augmented)) #x_train, x_augmented를 뒤에 엮겠다.
y_train = np.concatenate((y_train,y_augmented), axis=0)  #y는 scale하면 안됨!!!
x_test = x_test/255.
print(x_train.shape, y_train.shape) #(100000, 28, 28, 1), (100000)
print(np.max(x_train), np.min(x_train))         
print(np.max(x_augmented), np.min(x_augmented)) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#[실습] x_augmented 10개와 x_train 10개를 비교하는 이미지를 출력하기 (2,10)짜리

import matplotlib.pyplot as plt
plt.figure(figsize=(50,50)) #그림사이즈 
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')
    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()


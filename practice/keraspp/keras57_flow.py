#flow_from_directory
#flow

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
#seed고정
np.random.seed(42)


(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

#증폭
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 100

#랜덤하게 증폭
randx = np.random.randint(x_train.shape[0], size=augment_size) #(60000,100) : 랜덤하게 6만개 데이터중 100개 뽑음
print(np.min(randx), np.max(randx))

#증폭값 중복방지를 위해 변환 
x_aug = x_train[randx].copy()
y_aug = y_train[randx].copy()
print(x_aug.shape, y_aug.shape) #(100, 28, 28) (100,)

#증폭한 데이터 4차원으로 
x_train = x_train.reshape(-1,28,28,1)
x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_aug = x_aug.reshape(-1,28,28,1)

x_aug = train_datagen.flow(
    x_aug,y_aug, batch_size=augment_size, shuffle=False
).next()[0]

print(x_aug)
print(x_aug.shape) #(100, 28, 28, 1)

#scale & concat
x_train = np.concatenate((x_train/255., x_aug))
y_train = np.concatenate((y_train,y_aug), axis=0)
x_test = x_test/255.
print(x_train.shape, y_train.shape) #(60100, 28, 28, 1) (60100,)



'''
augment_size = 100
# a = np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1)
# print(a.shape) #(100,28,28,1)

# b = np.zeros(augment_size)
# print(b.shape) #(100,)
#1.
x_data = train_datagen.flow(
    a, b, batch_size=augment_size,shuffle=True,)
print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000001A2DEDA8520>
print(x_data[0]) #x,y
print(x_data[0][0].shape) #(100, 28, 28, 1) : x
print(x_data[0][1].shape) #(100,)           : y


#2.
x_data = train_datagen.flow(
    a, b, batch_size=augment_size,shuffle=True,).next()
print(type(x_data)) #<class 'tuple'> =>tuple(x,y)은 numpy구조(x:nump,y:nump)를 포함함(shape가능)
print(x_data)  #x,y
print(x_data[0])        
print(x_data[0].shape) #(100, 28, 28, 1) : x
print(x_data[1].shape) #(100,)           : y

#그래프
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][0][i], cmap='gray')
# plt.show()
'''


#그래프 
import matplotlib.pyplot as plt
plt.figure(figsize=(30,30)) #화면사이즈 
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')
    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.imshow(x_aug[i], cmap='gray')
plt.show()

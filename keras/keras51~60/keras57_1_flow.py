from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
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


argment_size = 100 #증폭사이즈 
print(np.tile(x_train[0].reshape(28*28),argment_size).reshape(-1,28,28,1).shape)
#np.tile(데이터, 증폭시킬 개수) 
#(100, 28, 28, 1) #1개의 이미지를 100개로 증폭시켜라 

print(np.zeros(argment_size))        #100개의 0을 출력해줌 
print(np.zeros(argment_size).shape)  #(100,)



# 디렉토리 있는 것을 이미지 만드는 것 : 경로 받아들임
# flow : 원래 있는 데이터를 증폭시키는 것

x_data = train_datagen.flow(
    np.tile(x_train[5].reshape(28*28),argment_size).reshape(-1,28,28,1), #x데이터
    np.zeros(argment_size), #y데이터(임의'0') : 그림만 그릴거라서 y값 필요없어서 임의의 숫자 0 넣음
    batch_size=argment_size,
    shuffle=True,
)

print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000001D31663FDF0>
print(x_data[0]) #x와 y가 모두 포함 
print(x_data[0][0].shape) #x : (100,28,28,1)
print(x_data[0][1])       #y : (100,)



import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()


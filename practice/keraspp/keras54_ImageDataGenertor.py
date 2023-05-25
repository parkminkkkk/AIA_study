import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#이미지전처리 
train_dtgen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    heigth_shift_range=0.1,
    rotation_range=5
)
test_dtgen = ImageDataGenerator(rescale=1./255)

#D드라이브에서 데이터 가져오기
xy_train = train_dtgen.flow_from_directory(
    'd:/study/',
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_dtgen.flow_from_directory(
    'd:/study/',
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)


#model
from tensorflow.keras.models import Sequential
model = Sequential()

#compile,fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit_generator(xy_train, epochs=10,
                    steps_per_epoch=32,
                    validation_data=xy_test,
                    validation_steps=24 )

hist = model.fit(xy_train, epochs=10,
                steps_per_epoch=32,
                validation_data=xy_test,
                validation_steps=24 )

#plot
import matplotlib.pyplot as plt 
import matplotlib

plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(hist.history['loss'], marker='.', label='acc', c='red')
plt.legend()
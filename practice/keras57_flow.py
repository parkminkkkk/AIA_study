#flow_from_directory
#flow

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
import autokeras as ak
from keras.datasets import mnist
print(ak.__version__)

import tensorflow as tf
import time


#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = \
#                             tf.keras.datasets.mnist.load_data()


#2. 모델 
path = "./_save/autokeras/"
model = tf.keras.models.load_model(path + "keras62_autokeras1.h5")


###. 최적의 모델 출력 
print(model.summary())



#4. 평가, 예측 
y_predict = model.predict(x_test)
# results = model.evaluate(x_test, y_test)
# print('model 결과 :', results)


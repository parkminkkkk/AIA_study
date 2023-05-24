import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(337)   #초기 가중치 고정 

#1. 데이터 
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 
model = Sequential()
model.add(Dense(3, input_dim =1))
# numpy=array([[ 0.5376226 , -0.7160384 , -0.19092572]] : 초기 weight 임의의 값 // [0., 0., 0.] : bias의 초기값 // 'dense/kernel:0' : kernel(layer상의 커널) = weight 
model.add(Dense(2))
model.add(Dense(1))
model.summary()

########################################################
model.trainable = False   # ★★★   /// 디폴트 : True
########################################################

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y,batch_size = 1, epochs=50)

y_predict = model.predict(x)
print(y_predict)

#


# False
# 5/5 [==============================] - 8s 4ms/step - loss: 7.2698  : 로스 고정됨 = 가중치 갱신 안되고있다는 뜻 
# [[0.1870494]
#  [0.3740988]
#  [0.5611482]
#  [0.7481976]
#  [0.9352468]]
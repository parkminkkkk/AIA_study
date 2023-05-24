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

print(model.weights)

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-1.175985  , -0.20179522, -1.1501358 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[ 0.5941951 , -0.95484424],[-0.62158144,  0.01421046],[-0.9479393 ,  0.9013896 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[0.23635006],[0.77884877]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print("==================================")
print(model.trainable_weights)  #model.weights와 동일함 
print("==================================")
print(len(model.weights))               # 6 : hidden layer개수 x 2 = 2(weight와 bias의 개수)
print(len(model.trainable_weights))     # 6 : 즉, hidden layer 하나당, 2개씩 생성된다 


#------------------------------------------------------------------------------------------------------#
model.trainable = False   # ★★★

print(len(model.weights))              # 6
print(len(model.trainable_weights))    # 0   
print("==================================")
print(model.trainable_weights)         # 정상적으로 weight값 들어가 있음 
print("==================================")
print(model.trainable_weights)         #[] : 빈값 

model.summary()
'''
=================================================================
Total params: 17
Trainable params: 0
Non-trainable params: 17   ***
_________________________________________________________________
'''


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet',  #가중치는 이미지넷에서 가져다 사용
              include_top=False,    #include_top: (True, False), Classification Layer의 포함여부 // False : input, output(fc_dense) layer제거 (Classification Layer를 제거)
              input_shape=(32,32,3)
              ) 

vgg16.trainable = False  #False : vgg16의 가중치 동결 


model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.trainable = True   ##vgg16만 가중치 동결(가져온 모델은 가중치 동결하고, 밑에 새로만든 dense는 가중치 형성해줌) 

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

################### 75_2번 소스에서 추가 ############################################################################
# print(model.layers)
'''
[<keras.engine.functional.Functional object at 0x000001B307388280>, 
<keras.layers.core.flatten.Flatten object at 0x000001B3073880D0>, 
<keras.layers.core.dense.Dense object at 0x000001B3073D42B0>, 
<keras.layers.core.dense.Dense object at 0x000001B3073D4490>]
'''

import pandas as pd 
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
# print(layers)
'''
  pd.set_option('max_colwidth', -1)
[(<keras.engine.functional.Functional object at 0x000001FC08F585B0>, 'vgg16', False), 
(<keras.layers.core.flatten.Flatten object at 0x000001FC08F58100>, 'flatten', True), 
(<keras.layers.core.dense.Dense object at 0x000001FC08F9C370>, 'dense', True), 
(<keras.layers.core.dense.Dense object at 0x000001FC08F9C490>, 'dense_1', True)]
'''
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
'''
                                                         Layer Type  Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x00000230170B71F0>  vgg16      False
1  <keras.layers.core.flatten.Flatten object at 0x00000230170B7430>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x0000023017108880>       dense      True
3  <keras.layers.core.dense.Dense object at 0x0000023017108760>       dense_1    True
'''








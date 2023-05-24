#layer 한개, 한개 가중치 동결 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1, name= 'hidden1'))
model.add(Dense(2, name= 'hidden2'))
model.add(Dense(1, name= 'outputs'))

#1. 전체 동결
# model.trainable = False   

#2. 전체동결 
# for layer in model.layers: 
#     layer.trainable = False

#3. 부분 layers 동결 
# print(model.layers[0])   #<keras.layers.core.dense.Dense object at 0x0000021440427F40>

model.layers[0].trainable = False   #hidden1
# model.layers[1].trainable = False   #hidden2
# model.layers[2].trainable = False   #outputs



model.summary()

#layer별 확인 
import pandas as pd 
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
# print(layers)
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

'''
#1.2. 전체 동결
                                                     Layer Type Layer Name  Layer Trainable
0  <keras.layers.core.dense.Dense object at 0x0000021427507F40>  hidden1    False
1  <keras.layers.core.dense.Dense object at 0x0000021434DAA820>  hidden2    False
2  <keras.layers.core.dense.Dense object at 0x0000021434EF4D60>  outputs    False

#3. 부분 layers 동결 
                                                     Layer Type Layer Name  Layer Trainable
0  <keras.layers.core.dense.Dense object at 0x000001E837997F40>  hidden1    False
1  <keras.layers.core.dense.Dense object at 0x000001E845236880>  hidden2    True
2  <keras.layers.core.dense.Dense object at 0x000001E8032D5DC0>  outputs    True


'''
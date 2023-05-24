import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

# model = VGG16()       #디폴트 :  include_top=True,  input_shape=(224,224,3)
model = VGG16(weights='imagenet',  #가중치는 이미지넷에서 가져다 사용
              include_top=False,    #include_top: (True, False), Classification Layer의 포함여부 // False : input, output(fc_dense) layer제거 (Classification Layer를 제거)
              input_shape=(32,32,3)
              ) 

model.summary()

print(len(model.weights))            #32 = 16 X 2   /// ==>>26 = 13 X 2  : fc (Dense) 부분 모두 없어짐 (3개의 dense layer)
print(len(model.trainable_weights))  #32 = 16 X 2  /// ==>>26  = 13 X 2  : fc (Dense) 부분 모두 없어짐 


######### include_top=True ########################################
# 디폴트 
#1. FC layer 원래 것 사용 
#2. input_shape=(224,224,3) 고정값 : 바꿀 수 없음 
'''
Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0

 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792   
 .
 .
 .
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0
 flatten (Flatten)           (None, 25088)             0

 fc1 (Dense)                 (None, 4096)              102764544

 fc2 (Dense)                 (None, 4096)              16781312

 predictions (Dense)         (None, 1000)              4097000

=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
'''
###################################################################


######### include_top=False ########################################
# 변경가능 
#1. FC layer 원래 것 삭제 => 나중에 커스터마이징 
#2. input_shape = (32,32,3) 변경가능 => 커스터마이징 

'''
Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 32, 32, 3)]       0

 block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
 .
 .
 .
 
 block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0   

 # Flatten하단부분 제거됨 
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
'''
####################################################################



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


# model.trainable = True일때 
# 30
# 30 

# model.trainable = False일때 
# 30
# 0


#vgg16.trainable = False일때 
# 30
# 4         # dense부분만 가중치 줌 (vgg16부분만 가중치 동결)


## vgg16의 가중치 동결하고 훈련 시키는 것이 성능 더 좋을 것이다../ 예외 존재(둘다 가중치 사용했을때, 성능 더 좋은 경우도 발생함 )

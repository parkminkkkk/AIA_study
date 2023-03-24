#(2X1),(1X2)같은 경우는 shape가 너무 작으니까 문제 발생 할 수 있음 
#데이터(shape)의 크기를 키움(padding) -> (값에 영향을 미치면 안되니까 임의의 값 0) -> 파라미터 수정

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten  #Convolution2D 동일 / 이미지CNN  #Flatten： 펼치기(평탄화)

model = Sequential()
model.add(Conv2D(7,(3,3),
                 padding='same',
                 input_shape=(8,8,1))) 
model.add(Conv2D(filters=4, kernel_size=(3,3), 
                 padding='valid',
                 activation='relu')) 
model.add(Conv2D(10, (2,2))) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))  #다중분류 함수

model.summary()

#padding default : 'valid' 
#padding 적용 : 'same'


#출력 (None, 7, 7, 7) -> padding'same'  -> 출력 (None, 8, 8, 7) : shape 유지 (padding적용으로 shape커져서 유지됨)
#출력 (None, 5, 5, 4) -> padding'valid' -> 출력 (None, 6, 6, 4) : shape 유지 안하고 싶을때 사용
#커널사이즈[kernel_size=(3X3)/(4X4)..]가 더 클 경우에는, padding을 더 크게 만들어서 shape 유지함
#즉, 커널사이즈에 맞춰서 padding사이즈를 맞춰준다. 

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 8, 8, 7)           35            #padding='same'
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 6, 6, 4)           256           #padding='valid' (default)
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 5, 10)          170
_________________________________________________________________
flatten (Flatten)            (None, 250)               0
_________________________________________________________________
dense (Dense)                (None, 32)                8032
_________________________________________________________________
dense_1 (Dense)              (None, 10)                330
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 33
=================================================================
Total params: 8,856
Trainable params: 8,856
Non-trainable params: 0
_________________________________________________________________
'''
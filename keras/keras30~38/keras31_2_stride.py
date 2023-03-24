#stride 커널사이즈의 보폭 , 디폴트값=1 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten  #Convolution2D 동일 / 이미지CNN  #Flatten： 펼치기(평탄화)

model = Sequential()
model.add(Conv2D(7,(2,2),
                 padding='same',
                 strides=2,
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

# model.add(MaxPooling2D()) 디폴트(2,2)임 = stride=2랑 동일 
# 커널사이즈보다 더 크게 stride를 주지 않음 / 왜냐하면 커널사이즈의 보폭이므로 그 이상을 주지 않는다

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten  #Convolution2D 동일 / 이미지CNN  #Flatten： 펼치기(평탄화)

model = Sequential()
model.add(Conv2D(7,                    #filters개수(output개수)/ (파라미터 튜닝)
                 (2,2),                #5x5이미지를 2x2로 자름 => 4x4로 바뀜 / 즉, 자르는 크기 '2x2' : 파라미터(수정가능)
                 input_shape=(8,8,1))) #'가로'5, '세로'5 이미지'흑백'1/''컬러'3 

#출력 : (N, 7, 7, 7) = (batch_size, rows, colums, channels) / # 출력 (batch_shape, new_rows, new_cols, filters)
#-batch_size :훈련의 수이므로, 여기서는 곧 filters개수가 훈련의 수임
#-channels :이미지'흑백'1/''컬러'3 
#현재 (8,8,1)의 데이터임 ->(한 레이어 통과한 후)-> (7,7,7)로 나옴 : 7x7, 7장
model.add(Conv2D(filters=4, 
                 kernel_size=(3,3), 
                 activation='relu')) 
#출력 : (N, 5, 5, 4) / 4차원 데이터를 받아서 출력 또한 4차원데이터 
model.add(Conv2D(10, (2,2))) 
#출력 : (N, 4, 4, 10) -> Flatten : 160개 데이터 
model.add(Flatten()) #출력 : (N, 4*4*10) = (N, 160)
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))  #다중분류 함수
#(N, 160) 2차원 데이터 됐으니까 이제 Dense 해줄 수 있음


model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 7, 7, 7)           35
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 5, 5, 4)           256
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 10)          170
_________________________________________________________________
flatten (Flatten)            (None, 160)               0
=================================================================
Total params: 461
Trainable params: 461
Non-trainable params: 0
_________________________________________________________________
'''
#=>특성은 줄었지만 가치는 높아짐(양은 더 많이 늘어남)














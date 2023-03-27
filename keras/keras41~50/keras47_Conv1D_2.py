import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Bidirectional, Dropout, Flatten

#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


# [실습] 80 만들기  

print(x.shape, y.shape) #(13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape) #(13,3,1)

#2. 모델구성 
model = Sequential()
model.add(Conv1D(10, 7, padding='same', input_shape=(3,1))) 
model.add(Conv1D(filters=5, kernel_size=(2), 
                 padding='valid',
                 activation='relu')) 
model.add(Conv1D(10, (2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()
#모델 로드
# model = load_model('./_save/MCP/keras40_save_model.h5') 
# model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)

#모델 저장
# model.save('./_save/MCP/keras40_save_model.h5') 


#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array([50,60,70]).reshape(1,3,1) 
print(x_predict.shape) #(1, 3, 1)


result = model.predict(x_predict)
print('loss:', loss)
print('[50,60,70]의 결과:', result)


'''
*Bidirectional
loss: 0.0006137349409982562
[50,60,70]의 결과: [[81.54205]]

*Conv1D_flatten(x)
loss: 493.1248474121094
[50,60,70]의 결과: [[[25.372723]]]

*Conv1D
loss: 1.1137807369232178
[50,60,70]의 결과: [[82.85317]]
'''
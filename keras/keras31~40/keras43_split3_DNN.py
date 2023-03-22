import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping



dataset = np.array(range(1, 101))   # 1~100
timesteps = 5                       # 5개씩 자르기 
x_predict = np.array(range(96,106)) # 100~106 예상값 [100:107] / # slice해주기

def split_x(dataset, timesteps):                   
    aaa = []                                      
    for i in range(len(dataset) - timesteps +1): 
        subset = dataset[i : (i + timesteps)]     
        aaa.append(subset)                         
    return np.array(aaa)                          

bbb = split_x(dataset, timesteps)
print(bbb)           
print(bbb.shape)  #(6, 5)


#방법2.
bb = timesteps-1
x = bbb[:, :bb] 
y = bbb[:, -1] 
x_predict = split_x(x_predict,bb)

print(x)
print(y)
print(x_predict)
print(x.shape, y.shape, x_predict.shape) #(96, 4) (96, 1) (7, 4)


#[실습]reshape하지 않고 dnn모델 구성
#2. 모델구성 
model = Sequential()
model.add(Dense(16, input_shape=(4,), activation='linear'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=10, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x, y, epochs=1000, callbacks=(es))

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array(x_predict)
print(x_predict.shape) 

result = model.predict(x_predict)
print('loss:', loss)
print('[100:107]의 결과:', result)

'''
1.LSTM
loss: 0.00018594646826386452
[100:107]의 결과: [[100.045456]
 [101.05736 ]
 [102.07036 ]
 [103.08448 ]
 [104.099785]
 [105.116295]
 [106.13402 ]]

2.DNN
 loss: 0.00012747147411573678
[100:107]의 결과: [[100.0126  ]
 [101.01677 ]
 [102.02153 ]
 [103.026955]
 [104.03302 ]
 [105.03976 ]
 [106.04723 ]]

'''
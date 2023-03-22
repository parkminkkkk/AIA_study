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

x = bbb[:, :4]    #[행,열(처음(0)~4'전'까지)]
# x = bbb[:, :-1] #[모든행, 마지막 열 '전'까지]
y = bbb[:, -1]    #[모든행, 마지막 열]

x_predict = split_x(x_predict, 4)

print(x)
print(y)
print(x_predict)
print(x.shape, y.shape, x_predict.shape) #(96, 4) (96, 1) (7, 4)

x= x.reshape(96,4,1)
x_predict= x_predict.reshape(7,4,1) 

print(x.shape, x_predict.shape)   #(96, 4, 1) (7, 4, 1)

'''
방법2.
bb = timesteps-1
x = bbb[:, :bb] 
y = bbb[:, -1] 
x_predict = split_x(x_predict,bb)
'''

'''
###위에 함수를 명시했기때문에 다시 한번 더 할 필요 xx###
#            #(x_pred, 4)
# def split_x(x_predict, timesteps = 4):                   
#     aaa = []                                      
#     for i in range(len(x_predict) - timesteps +1): 
#         subset = x_predict[i : (i + timesteps)]     
#         aaa.append(subset)                         
#     return np.array(aaa)                          
# ccc = split_x(x_predict, timesteps=4)
# print(ccc)          
'''


#2. 모델구성 
model = Sequential()
model.add(LSTM(16, input_shape=(4,1), activation='linear')) #[batch, / timesteps, feature]   
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(8, activation='relu'))
model.add(Dense(1)) 

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='acc', patience=10, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit(x, y, epochs=1000, callbacks=(es))

#4. 평가, 예측 
loss = model.evaluate(x, y)
x_predict = np.array(x_predict).reshape(7,4,1) #[[[8],[9],[10]]] #스칼라3개, 벡터1개 =1차원 -> 3차원으로 바꿔줘야함(데이터 1개, input_shape맞춰줌(3,1))
print(x_predict.shape) #(1, 4, 1)

result = model.predict(x_predict)
print('loss:', loss)
print('[100:107]의 결과:', result)

'''
loss: 0.00018594646826386452
[100:107]의 결과: [[100.045456]
 [101.05736 ]
 [102.07036 ]
 [103.08448 ]
 [104.099785]
 [105.116295]
 [106.13402 ]]

'''
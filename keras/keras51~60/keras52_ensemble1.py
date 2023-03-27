#앙상블 : 모델 합침
#앙상블기법 중 하나(stacking) : 대회때 test데이터 셋을 먼저 예측한 다음 다시 train 데이터에 넣어서 모델을 다시 돌려서 사용함
#(똑같은 모델 사용시 과적합 걸릴 수 있어서 다른 모델 사용)

#1. 데이터 

import numpy as np
x1_datasets = np.array([range(100), range(301,401)])    # 삼성, 아모레 
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) # 온도, 습도, 강수량 
print(x1_datasets.shape) #(2, 100)
print(x2_datasets.shape) #(3, 100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
print(x1.shape) #(100, 2)
print(x2.shape) #(100, 3)

y = np.array(range(2001,2101))  # 환율


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, random_state=640874
)
#train_test_split 3개도 가능
print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) #(70, 3) (30, 3)
print(y_train.shape, y_test.shape)   #(70,) (30,)

#2. 모델구성 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1.
input1 = Input(shape=(2,))
dense1 = Dense(16, activation='relu', name='stock1')(input1)
dense2 = Dense(24, activation='relu', name='stock2')(dense1)
dense3 = Dense(32, activation='relu', name='stock3')(dense2)
output1 = Dense(16, activation='relu', name='output1')(dense3)  
#output layer도 히든레이어로 여겨지기때문에 굳이 output nods를 1개 줄 필요 없다! 오히려 합병한 이후 소멸 될 수 있으므로 노드의 개수을 주는 것이 좋다

#2-2 모델2. 
input2 = Input(shape=(3,))
dense11 = Dense(16, activation='relu', name='weather1')(input2)
dense12 = Dense(16, activation='relu', name='weather2')(dense11)
dense13 = Dense(32, activation='swish', name='weather3')(dense12)
dense14 = Dense(16, activation='swish', name='weather4')(dense13)
output2 = Dense(16, name='output2')(dense14)

#2-3 모델 합침
from tensorflow.keras.layers import concatenate, Concatenate  #소문자(함수) #대문자(클래스) 
merge1 = concatenate([output1, output2], name='mg1')  # 두 모델의 아웃풋을 합병한다(concatenate모델의 input은 1,2모델의 아웃풋을 합친 것). #2개 이상은 list형태로 받음
merge2 = Dense(24, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-4 모델 정의 
model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

'''
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_2 (InputLayer)           [(None, 3)]          0           []

 input_1 (InputLayer)           [(None, 2)]          0           []

 weather1 (Dense)               (None, 10)           40          ['input_2[0][0]']

 stock1 (Dense)                 (None, 10)           30          ['input_1[0][0]']

 weather2 (Dense)               (None, 10)           110         ['weather1[0][0]']

 stock2 (Dense)                 (None, 20)           220         ['stock1[0][0]']

 weather3 (Dense)               (None, 10)           110         ['weather2[0][0]']

 stock3 (Dense)                 (None, 30)           630         ['stock2[0][0]']

 weather4 (Dense)               (None, 10)           110         ['weather3[0][0]']

 output1 (Dense)                (None, 11)           341          ['stock3[0][0]']

 output2 (Dense)                (None, 11)           121          ['weather4[0][0]']     #output layer도 히든레이어로 여겨지기때문에 굳이 output nods를 1개 줄 필요 없다! 오히려 합병한 이후 소멸 될 수 있으므로 노드의 개수을 주는 것이 좋다

 mg1 (Concatenate)              (None, 22)           0           ['output1[0][0]',      #연산량'0' 단순히 붙이기만 함  #(None, 22):'22'= output1(11) + output2(11)
                                                                  'output2[0][0]']

 mg2 (Dense)                    (None, 2)            46           ['mg1[0][0]']

 mg3 (Dense)                    (None, 3)            9           ['mg2[0][0]']

 last (Dense)                   (None, 1)            4           ['mg3[0][0]']

==================================================================================================
Total params: 1,311
Trainable params: 1,311
Non-trainable params: 0
__________________________________________________________________________________________________
'''

#3. 컴파일, 훈련 

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit([x1_train, x2_train],y_train, epochs=500, batch_size=16, validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측 

loss = model.evaluate([x1_test, x2_test], y_test)
print("loss:", loss)

predict = model.predict([x1_test, x2_test])
# print("predict", predict)

#r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
r2 = r2_score(y_test, predict)
print("r2:", r2)

#rmse
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, predict)
print("rmse:", rmse)


'''
loss: [0.029226841405034065, 0.029226841405034065]
r2: 0.9999673765647006
rmse: 0.17097816859882414

loss: [0.036939818412065506, 0.15081380307674408]
r2: 0.9999587696969646
rmse: 0.19221358496984134
'''
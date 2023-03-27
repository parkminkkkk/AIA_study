#1. 데이터 

import numpy as np
x1_datasets = np.array([range(100), range(301,401)])    # 삼성, 아모레 
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) # 온도, 습도, 강수량
x3_datasets = np.array([range(201,301), range(511,611), range(1300,1400)])  

print(x1_datasets.shape) #(2, 100)
print(x2_datasets.shape) #(3, 100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T
print(x1.shape) #(100, 2)
print(x2.shape) #(100, 3)
print(x3.shape) #(100, 3)


y = np.array(range(2001,2101))  # 환율

###random_state맞춰줘야 동일하게 잘림###
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, train_size=0.7, random_state=640874
)


#train_test_split 3개도 가능
print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) #(70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) #(70, 3) (30, 3)
print(y_train.shape, y_test.shape)   #(70,) (30,)

#2. 모델구성 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1.
input1 = Input(shape=(2,))
dense01 = Dense(16, activation='relu', name='stock1')(input1)
dense02 = Dense(24, activation='relu', name='stock2')(dense01)
dense03 = Dense(32, activation='relu', name='stock3')(dense02)
output1 = Dense(16, activation='relu', name='output1')(dense03)  
#output layer도 히든레이어로 여겨지기때문에 굳이 output nods를 1개 줄 필요 없다! 오히려 합병한 이후 소멸 될 수 있으므로 노드의 개수을 주는 것이 좋다

#2-2 모델2. 
input2 = Input(shape=(3,))
dense11 = Dense(16, activation='relu', name='weather1')(input2)
dense12 = Dense(16, activation='relu', name='weather2')(dense11)
dense13 = Dense(32, activation='swish', name='weather3')(dense12)
dense14 = Dense(16, activation='swish', name='weather4')(dense13)
output2 = Dense(16, name='output2')(dense14)

#2-3 모델3. 
input3 = Input(shape=(3,))
dense21 = Dense(16, activation='relu', name='s1')(input3)
dense22 = Dense(16, activation='relu', name='s2')(dense21)
dense23 = Dense(32, activation='swish', name='s3')(dense22)
dense24 = Dense(16, activation='swish', name='s4')(dense23)
output3 = Dense(16, name='output3')(dense24)

#2-4 모델 합침
from tensorflow.keras.layers import concatenate, Concatenate  #소문자(함수) #대문자(클래스) 
merge1 = concatenate([output1, output2, output3], name='mg1')  # 두 모델의 아웃풋을 합병한다(concatenate모델의 input은 1,2모델의 아웃풋을 합친 것). #2개 이상은 list형태로 받음
merge2 = Dense(24, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-4 모델 정의 
model = Model(inputs=[input1, input2, input3], outputs=last_output)

# model.summary()

#3. 컴파일, 훈련 

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit([x1_train, x2_train, x3_train],y_train, epochs=500, batch_size=16, validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측 

loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print("loss:", loss)

predict = model.predict([x1_test, x2_test, x3_test])
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
loss: [0.012731335125863552, 0.012731335125863552]
r2: 0.9999857984595405
rmse: 0.11280894641087323

loss: [0.0018325065029785037, 0.03001708909869194]
r2: 0.9999979547444398
rmse: 0.042810431613752746
'''
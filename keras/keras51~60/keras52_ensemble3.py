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


#마지막 아웃풋 노드 1개짜리가 2개 나와야함
y1 = np.array(range(2001,2101))  # 환율
y2 = np.array(range(1001,1101))  # 금리



###random_state맞춰줘야 동일하게 잘림### 
###'\' = 한번 잘라주기 (파이썬)###
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, x3, y1, y2, train_size=0.7, random_state=640874)

#train_test_split 5개도 가능
print(x1_train.shape, x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) #(70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) #(70, 3) (30, 3)
print(y1_train.shape, y1_test.shape)   #(70,) (30,)
print(y2_train.shape, y2_test.shape)   #(70,) (30,)

#2. 모델구성 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#======X모델 구성============================================
#2-1. 모델1.
input1 = Input(shape=(2,))
dense01 = Dense(16, activation='relu', name='stock1')(input1)
dense02 = Dense(24, activation='relu', name='stock2')(dense01)
dense03 = Dense(32, activation='relu', name='stock3')(dense02)
output1 = Dense(16, activation='relu', name='output1')(dense03)  

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

#======concatenate============================================
#2-4 모델 합침
from tensorflow.keras.layers import concatenate, Concatenate  #소문자(함수) #대문자(클래스) 
merge1 = concatenate([output1, output2, output3], name='mg1')  # 두 모델의 아웃풋을 합병한다(concatenate모델의 input은 1,2모델의 아웃풋을 합친 것). #2개 이상은 list형태로 받음
merge2 = Dense(24, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
hidden_output = Dense(16, name='hidden1')(merge3)

#======Y모델 구성(분기점)===============================================
#2-5. 분기점 모델1
bungi1 = Dense(10, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, activation='selu', name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)


#2-6. 분기점 모델2
bungi11 = Dense(10, name='bg11')(hidden_output)
bungi12 = Dense(10, name='bg12')(bungi11)
bungi13 = Dense(10, name='bg13')(bungi12)
last_output2 = Dense(1, name='last2')(bungi13)


#2-7 모델 정의 
model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])

# model.summary()


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

model.fit([x1_train, x2_train, x3_train],[y1_train,y2_train], epochs=500, batch_size=16, validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측 

loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("loss합:", loss[0])
print("last1_loss:", loss[1])
print("last2_loss:", loss[2])


y_predict = model.predict([x1_test, x2_test, x3_test])
# print("predict", y_predict)
# print(len(y_predict), len(y_predict[0]))  #2,30 #y가 몇개인지, y의 행 개수 
##list형태는 shape안됨(파이썬) // np.numpy(y_predict)해주면 shape가능


#r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print("r2_1:", r2_1)
print("r2_2:", r2_2)
print("r2스코어", (r2_1+r2_2)/2)


#rmse
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse1 = RMSE(y1_test, y_predict[0])
rmse2 = RMSE(y2_test, y_predict[1])
print("rmse:", rmse1)
print("rmse:", rmse2)


'''
loss: [171.20286560058594, 37.04472351074219, 134.15814208984375, 5.305981636047363, 10.249975204467773]
      #loss1,loss2의 합     #last1_loss        #last2_loss         #last1_mae          #last2_mae
r2_1: 0.9586594407011642
r2_2: 0.8502849662683445
rmse: 6.086449028957183
rmse: 11.582665693897894


loss합: 0.18560756742954254
last1_loss: 0.12649250030517578
last2_loss: 0.05911507084965706

r2_1: 0.9998588197088557
r2_2: 0.9999340299787818
r2스코어 0.9998964248438187  #(2개 합/2)

rmse: 0.35568278998491376
rmse: 0.24313592147894594
'''
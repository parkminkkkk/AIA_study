#저장방식 save_model과 비교 

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=1234)

#data scaling(스케일링)
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성 (함수형모델) 
input1 = Input(shape=(13,), name='h1') 
dense1 = Dense(30, name='h2')(input1) 
dense2 = Dense(20, name='h3', activation='relu')(dense1)                                                            
dense3 = Dense(10, name='h4',activation='relu')(dense2)     
output1 = Dense(1, name='h5',)(dense3)
model = Model(inputs=input1, outputs=output1) 


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                   verbose=1, 
                #    restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath='./_save/MCP/keras27_MCP.hdf5') #파일저장1
 
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, 
          callbacks=[es, mcp])


model.save('./_save/MCP/keras27_3_save_model.h5') #파일저장2 : 2개의 가중치와 모델 저장된 파일 형성

#파일 2개 저장되므로, MCP파일, save_model파일 두개 비교가능 


#4. 평가, 예측 
from sklearn.metrics import r2_score

print("==================== 1. 기본출력 =====================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)

print("==================== 2. load_model 출력 =====================")
model2 = load_model('./_save/MCP/keras27_3_save_model.h5')  #save_model 출력 
loss = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)

print("==================== 3. MCT 출력 =====================")
model3 = load_model('./_save/MCP/keras27_MCP.hdf5')  #save_MCTmodel 출력 
loss = model3.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어', r2)



'''
*동일한 가중치라 동일한 값 나옴 
==================== 1. 기본출력 =====================
4/4 [==============================] - 0s 749us/step - loss: 22.0987     # evaluate에서 verbose=0 잡으면 없어짐 
loss : 22.098670959472656
r2스코어 0.7847157843946944
==================== 2. load_model 출력 =====================
loss : 22.098670959472656
r2스코어 0.7847157843946944
==================== 3. MCT 출력 =====================
loss : 20.09746742248535
r2스코어 0.8042114381289059

'''

# ModelCheckpoint : 가장 낮은 지점에서 저장 (지점마다 저장됨, 가장 낮은지점까지 덮어씀)
# EarlyStopping : 가장 낮은 지점에서 patience=10 만큼 밀림 
# restore_best_weights=True 주석처리하고 다시 돌리면 MCT랑 load_model 값 차이남 


'''
*restore_best_weights=True 주석처리하고 다시 돌리면 MCT랑 load_model 값 차이남 
==================== 1. 기본출력 =====================
loss : 17.498199462890625
r2스코어 0.8295333654266284
==================== 2. load_model 출력 =====================
loss : 17.498199462890625
r2스코어 0.8295333654266284
==================== 3. MCT 출력 =====================
loss : 18.60220718383789
r2스코어 0.8295333654266284
'''


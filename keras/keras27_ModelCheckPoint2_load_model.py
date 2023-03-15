# ModelCheckPoint_load_model (import)

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
# print(type(x)) #<class 'numpy.ndarray'>
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=1234)

#data scaling(스케일링)
scaler = MinMaxScaler() 
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train) #위에 두줄 한줄로 합침 fit_transform
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test)) 


'''
#2. 모델구성 (함수형모델) 
input1 = Input(shape=(13,), name='h1') 
dense1 = Dense(30, name='h2')(input1) 
dense2 = Dense(20, name='h3', activation='relu')(dense1)                                                            
dense3 = Dense(10, name='h4',activation='relu')(dense2)     
output1 = Dense(1, name='h5',)(dense3)
model = Model(inputs=input1, outputs=output1) 

#모델 저장
# model.save('./_save/keras26_1_save_model.h5')


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                   verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', #val_loss낮은게 좋으니까 auto = min으로 적용
                      verbose=1, save_best_only=True,  #가장 좋은지점만 저장
                      filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5') #파일저장 
 
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, 
          callbacks=[es, mcp])
'''

model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5') 
#모델과 가중치 저장되어있음 (중간중간 저장할 수 있고, 몇epochs에서 저장할지 볼 수 있음)


#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


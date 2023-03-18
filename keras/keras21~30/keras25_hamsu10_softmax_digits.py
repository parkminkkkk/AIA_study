#사이킷런 load_digits

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score #분류=> 결과지표 'accuracy_score' 떠올라야함
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터 
datasets = load_digits()
# print(datasets.DESCR) #(1797, 64)  #pandas : describe()
# print(datasets.feature_names)  #pandas : colums()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(1797, 64) (1797,)
print(x)
print(y)  
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2 3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True))  

##########데이터 분리전에 one-hot encoding하기##########################
#y값 (1797,) ->  (1797,10) 만들어주기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape) #(1797, 10)
##################################################################


#데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y    
)
print(y_train)                                  
print(np.unique(y_train, return_counts=True))  

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() 
# scaler = StandardScaler() 
scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델구성
# model = Sequential()
# model.add(Dense(8, activation='relu', input_dim=64))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(10, activation='softmax')) #3개의 데이터를 뽑으니까 *label의 개수만큼 노드를 잡는다!! 

input1 = Input(shape=(64,))
dense1 = Dense(8,activation='relu')(input1)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(4, activation='relu')(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=3000, batch_size=16,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results)  
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_acc = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score:', acc)

'''
1. random_state=123, patience=100, epochs=3000, batch_size-16
results: [0.9030895829200745, 0.75] / accuracy_score: 0.75

2. *MinMaxScaler
results: [0.7188637852668762, 0.8444444537162781]/ accuracy_score: 0.8444444444444444
3. *RobstScaler 
Epoch 00147: early stopping/results: [0.8375341892242432, 0.7472222447395325]/accuracy_score: 0.7472222222222222
4. *StandardScaler 
Epoch 00193: early stopping/ results: [1.195489525794983, 0.7666666507720947]/ accuracy_score: 0.7666666666666667
5. *MaxAbsScaler 
Epoch 00153: early stopping/ results: [0.5333125591278076, 0.875]/ accuracy_score: 0.875

6.함수형모델
Epoch 00208: early stopping/ results: [0.5807938575744629, 0.8361111283302307]/ accuracy_score: 0.8361111111111111
'''
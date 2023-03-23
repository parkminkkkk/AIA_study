#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score

#1. 데이터 
datasets = load_iris()
# print(datasets.DESCR)
'''
#class : label값 (3개를 맞춰라)
- Iris-Setosa
- Iris-Versicolour
- Iris-Virginica
'''
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2]

#1-1)one-hot-encoding
#y값 (150, ) -> (150,3) 만들어주기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(150, 3)

#1-2)데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y)

#1-3)data scaling(스케일링)
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
# print(np.min(x_test), np.max(x_test)) 


#2. 모델구성 (함수형모델) 

input1 = Input(shape=(4,))
dense1 = Dense(8,activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(4, activation='relu')(drop1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(4, activation='relu')(dense3)
output1 = Dense(3, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)


# model = Sequential()
# model.add(Dense(8, input_shape=(4,)))
# model.add(Dropout(0.3))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#시간저장
import datetime 
date = datetime.datetime.now()  #현재시간 데이터에 넣어줌
print(date)  #2023-03-14 11:15:21.154501
date = date.strftime("%m%d_%H%M")  #'%'특수한 경우에 반환하라 -> month,day_Hour,Minute
#시간을 문자데이터로 바꿈 : 문자로 바꿔야 파일명에 넣을 수 있음 
print(date) #0314_1115

#경로명 
filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04 : 4번째자리, .4: 소수점자리 - hist에서 가져옴 



from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', 
                   verbose=1, 
                   restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', 
                      verbose=1, save_best_only=True,  
                      filepath="".join([filepath, 'k27_', date, '_', filename])
                      ) 
 
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, 
          callbacks=(es)) #[mcp])



#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print(results)

y_pred = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1) #각 행에 있는 '열(1)'끼리 비교 
y_pred = np.argmax(y_pred, axis=-1) #axis=-1, axis=1 모두 가능 / 만약 axis=0하면 (3,)나오므로(x)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score:', acc)

'''
3. 함수형모델
Epoch 00998: early stopping
[0.11794759333133698, 0.9333333373069763]
accuracy_score: 0.9333333333333333

4. *dropout 
[0.19856961071491241, 0.8666666746139526]
accuracy_score: 0.8666666666666667

5. 함수형, dropout 
[0.27433863282203674, 0.8999999761581421]
accuracy_score: 0.9
'''
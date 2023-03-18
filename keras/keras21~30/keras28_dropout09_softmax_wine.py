#과적합 제거(해결)
# 1. 데이터 많으면 됨 
# 2. 전체 중에서 일부 노드 빼고 훈련시킨다(dropout): 파라미터값 수정


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터 
datasets = load_wine()
# print(datasets.DESCR) #(150,4)  #pandas : describe()
'''
#class : label값 (3개를 맞춰라)
 - class_0
 - class_1
 - class_2
'''
# print(datasets.feature_names)  #pandas : colums()
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'] 


x = datasets.data
y = datasets['target']
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2]


#1-1)one-hot-encoding
#y값 (178, ) -> (178,3) 만들어주기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(178,3)

#1-2)데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y)
# print(y_train)                                  
# print(np.unique(y_train, return_counts=True))  

#1-3)data scaling(스케일링)
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
# print(np.min(x_test), np.max(x_test)) 


#2. 모델구성 (함수형모델) 
'''
input1 = Input(shape=(13,))
dense1 = Dense(8,activation='relu')(input1)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(4, activation='relu')(dense3)
output1 = Dense(3, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
'''

model = Sequential()
model.add(Dense(8, input_shape=(13,)))
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

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
print('results:', results)

y_predict = model.predict(x_test)
# print(y_predict.shape) #(36, 10)
y_predict = np.argmax(y_predict, axis=1) 
# print(y_predict.shape) #(36,)
#axis=1, axis=-1동일 (왜냐하면 one-hot하면 2차원으로 데이터가 나오니까 동일함)

# print(y_test.shape) #(36, 10)
y_test = np.argmax(y_test, axis=1)
# print(y_test.shape) #(36, )


from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

'''
3. 함수형모델
results: [0.012789257802069187, 1.0]
acc:  1.0
4. *dropout 
results: [0.008415436372160912, 1.0]
acc:  1.0
'''
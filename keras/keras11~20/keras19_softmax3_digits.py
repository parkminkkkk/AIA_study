#사이킷런 load_digits

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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


#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=64))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(10, activation='softmax')) #3개의 데이터를 뽑으니까 *label의 개수만큼 노드를 잡는다!! 

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
results: [0.9030895829200745, 0.75]
accuracy_score: 0.75
'''
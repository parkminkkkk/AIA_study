import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 

#1. 데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']


# 1-1 one-hot-encoding
#pandas get_dummies
import pandas as pd
y=pd.get_dummies(y)
# y = np.array(y) 
print(y.shape) #(581012, 7)

#1-2 데이터분리 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, train_size=0.8,stratify=y)


#2. 모델구성 
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=54))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.summary() #Total params: 587


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

#ES추가 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True, verbose=1)

#훈련시간 확인 
#epochs 작게 잡아서 돌려보고 예측할 수 있음 
import time
start_time = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es],
                 )
end_time = time.time()
# print("걸린시간(S) : ", end_time - start_time)
print("걸린시간(S) : ", round(end_time - start_time, 2)) #파이썬 자체에도 예약어 있음, round가능 (빨간줄 안뜨니까)


'''
#4. 평가, 예측 

results = model.evaluate(x_test, y_test)
print('results:', results)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) #print(y_pred.shape)
y_test_acc = np.argmax(y_test, axis=1) #print(y_test.shape)

acc = accuracy_score(y_test_acc, y_pred)
print('acc:', acc)
'''
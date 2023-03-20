#다중분류
# output layer, loss만 다름
#1. 마지막 아웃풋 레이어, activation = 'softmax'사용 (바뀌지않음!무조건)
#   y의 라벨값의 개수만큼 노드를 정해준다!(이중분류와의 차이점)
#2. loss = 'categorical_crossentropy'사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import accuracy_score 
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터 
datasets = load_iris()
print(datasets.DESCR) #(150,4)  #pandas : describe()
'''
#class : label값 (3개를 맞춰라)
- Iris-Setosa
- Iris-Versicolour
- Iris-Virginica
'''
print(datasets.feature_names)  #pandas : colums()
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(150, 4) (150,) 
print(x)
print(y)  #0,1,2로만 순서대로 나옴-> 섞어줘야함(shuffle)
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2]

##########데이터 분리전에 one-hot encoding하기##########################
#y값 (150, ) -> (150,3) 만들어주기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape) #(150, 3)
#판다스에 겟더미, 사이킷런에 원핫인코더 (과제)
##################################################################


#데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y    #통계적으로 (y값,같은 비율로)
)
print(y_train)                                  #[0 1 0 0 1 0 1 2 2 1 0 2 2 1 2]
print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2]), array([5, 5, 5], dtype=int64)) #return_counts=True : 개수 반환
#ex) 0:1:2=3:2:10일경우, train의 값이 '2'에 너무 치우쳐있어서 연산의 값이 '2'에 많이 치우칠 수 있음 
#즉, 훈련데이터가 특정값에 몰려있으면, 그 값이 나올 확률이 높다/ y의 라벨 숫자도 영향을 미칠수 있다
#따라서 y_label값 확인해야함/ y의 비율이 비슷해야함 (=train비율만큼 train_test_split도 같은 비율로 잘라줘야한다)

#data scaling(스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

#reshape
print(x_train.shape, x_test.shape) #(120, 4) (30, 4)
x_train= x_train.reshape(-1,4,1,1)
x_test= x_test.reshape(-1,4,1,1)

#2. 모델구성
# model = Sequential()
# model.add(Dense(8, activation='relu', input_dim=4))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='softmax')) 

model = Sequential()
model.add(Conv2D(8,(2,1),
                 padding='same',
                 input_shape=(4,1,1))) 
model.add(Conv2D(filters=5, kernel_size=(2,1), 
                 padding='valid',
                 activation='relu')) 
model.add(Conv2D(16, (2,1))) 
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=16,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

# *ValueError: Shapes (16, 1) and (16, 3) are incompatible 
# y의 라벨값의 개수 = y의 클래스의 개수
# (150,)인데 노드3으로 잡으면 (150,3)으로 출력하기때문에 오류가 발생 (..y값1개인데 이 안에 클래스3개 들어가있음..)
# => one-hot Encoding 사용!
#softmax의 전체의 합=1, one-hot encoding의 전체의 합 =1 


#4. 평가, 예측 
results = model.evaluate(x_test, y_test)
print(results) #[0.9782050251960754, 0.5666666626930237] : list형태 
# print('loss :', results[0])  #loss : 0.9942742586135864
# print('acc :', results[1])   #acc : 0.3333333432674408

y_pred = model.predict(x_test)
'''
print(y_test.shape) #(30,3)
print(y_pred.shape) #(30, 3) - softmax한 이후의 값 : 0.xxx값 나옴
print(y_test[:5])   #[1.0.0.]
print(y_pred[:5])   #[0.333. 0.331. 0.333] : softmax한 이후의 값
#따라서, y_test, y_pred 모두 argmax를 사용해서 가장 큰 수의 위치값으로 반환하기
#argmax사용해서 나온 값들끼리(위치값으로 표현된 데이터)비교해서 정확도를 확인 가능=> 'acc스코어'
'''

y_test_acc = np.argmax(y_test, axis=1) #각 행에 있는 '열(1)'끼리 비교 
y_pred = np.argmax(y_pred, axis=-1) #axis=-1, axis=1 모두 가능 / 만약 axis=0하면 (3,)나오므로(x)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score:', acc)


'''
1. *MinMaxScaler 
Epoch 00786: early stopping
[0.27919596433639526, 0.8333333134651184]
accuracy_score: 0.8333333333333334
2. 함수형모델
Epoch 00653: early stopping
[0.32986006140708923, 0.8333333134651184]
accuracy_score: 0.8333333333333334
3. 함수형모델
Epoch 00998: early stopping
[0.11794759333133698, 0.9333333373069763]
accuracy_score: 0.9333333333333333

*cnn
[0.9275529384613037, 0.6666666865348816]
accuracy_score: 0.6666666666666666
'''


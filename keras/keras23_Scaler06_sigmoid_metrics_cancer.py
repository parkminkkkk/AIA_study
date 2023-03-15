#이진분류 
#1. 마지막 아웃풋레이어, activation = 'sigmoid'사용
#2. loss = 'binary_crossentropy'사용
#값이 0과 1로만 나올 수 있게 해줘야함!

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터 
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)        #pandas : .describe()  / 실무에서는 pandas사용함* 
print(datasets.feature_names)  #pandas : .colums()

x = datasets['data']
y = datasets.target  #data, target은 'dictionary'의 '키' / print(datasets)해서 볼 수 있음

print(x.shape, y.shape)   #(569, 30) (569,) 
# print(y) # y에 들어가는 값이 0,1뿐 (분류)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=777, test_size=0.2,
)

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


#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='linear', input_dim=30))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #이진분류 - 마지막 아웃풋레이어, 'sigmoid'사용 (0,1사이로 한정) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',   #이진분류 - loss, 'binary_crossentropy'사용  
              metrics=['accuracy','mse'] 
              #['accuracy', 'acc', 'mse', 'mean_squared_error'] 가능 / cf) 'r2'는 metrics에 없음.. / metrics : 각 loss마다 acc,mse 보여줌
              #훈련과정에 accuracy 나옴/ list[] : 값을 리스트로 받는것은 2개이상 들어갈 수 있음
              ) 
#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=5000, batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results) 
#값이 여러개 나오므로 results : loss(binary_crossentropy), accuracy, mse 
#즉, loss와 metrics에 넣어놓은 값이 출력된다
 
y_predict = np.round(model.predict(x_test)) #np.round:반올림 / 예측값을 반올림해서 0,1의 값이 나올 수 있도록해줌 (0.5까지는 0으로, 0.6부터는 1로)
# print(y_predict)


from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)



'''
1. results: [0.13051943480968475, 0.9473684430122375, 0.039755672216415405]/ acc:  0.9473684210526315

2 *stratify=y(o) #같은비율일때, 통상적으로 더 잘 나옴
results: [0.06227986887097359, 0.9824561476707458, 0.015651952475309372]/ acc:  0.982456140350877
3. *stratify=y(x)
results: [0.12325089424848557, 0.9210526347160339, 0.041780561208724976]/ acc:  0.9210526315789473

4.*MinMaxScaler
Epoch 00248: early stopping/ results: [0.18867246806621552, 0.9473684430122375, 0.0516931377351284]/ acc:  0.9473684210526315
'''

#이진분류 
#1. 마지막 아웃풋레이어, activation = 'sigmoid'사용
#2. loss = 'binary_crossentropy'사용
#값이 0과 1로만 나올 수 있게 해줘야함!

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_breast_cancer()
print(datasets.feature_names)  #pandas : .colums()

x = datasets['data']
y = datasets.target 

print(x.shape, y.shape)   #(569, 30) (569,) 
# print(y) # y에 들어가는 값이 0,1뿐 (분류)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640874, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='linear', input_dim=30))
model.add(Dense(8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #이진분류 - 마지막 아웃풋레이어, 'sigmoid'사용 (0,1사이로 한정) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',   #이진분류 - loss, 'binary_crossentropy'사용  
              metrics=['accuracy','mse'] 
              ) 
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=25, mode='min', 
              verbose=1,
              restore_best_weights=True
             )  

model.fit(x_train, y_train, epochs=1000, batch_size=8,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results) 
 
y_predict = np.round(model.predict(x_test)) #np.round:반올림 / 예측값을 반올림해서 0,1의 값이 나올 수 있도록해줌 (0.5까지는 0으로, 0.6부터는 1로)

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)



'''
results: [0.13051943480968475, 0.9473684430122375, 0.039755672216415405]
acc:  0.9473684210526315

es
'''

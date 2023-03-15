#데이콘 따릉이 문제풀이(대회) *중요* 많이 써먹음!! 머리에 다 넣기!!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #r2스코어, mse 궁금함 -> 불러옴/ mse지표에 루트씌워서 rmse지표 만들어주기 
import pandas as pd #깔끔하게 데이터화 됨(csv 데이터 가져올때 좋음) *실무에서 엄청 씀 pandas*

#1. 데이터 
path = './_data/ddarung/' #'.'= 현재폴더(study)  '/'=하단, _data하단의 ddarung데이터 
# train_csv = pd.read_csv('./_data/ddarung/train.csv')  #원래 이렇게 써야함/ 자주쓰니까 path로 명명

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #0번째는 인덱스 칼럼이야  
print(train_csv) #확인
print(train_csv.shape) 

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0) 
print(test_csv)
print(test_csv.shape) #(715, 9) : count 없어서 (10->9)

#
datasets = train_csv()
x = datasets.test_csv
y = datasets.submission.csv

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성 
model = Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(16))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='rmse', optimizer='adam')
model.fit(x_train,y_train, epochs=10, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

'''
#뭐가 잘못?? 모르게써...
-> 데이터가 없어서 error
  File "c:\study\keras_mk\kerasmk13_ddarung.py", line 25, in <module>
    datasets = train_csv()
TypeError: 'DataFrame' object is not callable
'''
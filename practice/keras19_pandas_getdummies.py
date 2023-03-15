#판다스에 겟더미
#pandas get_dummies
#pandas => error발생 : (데이터종류) 코드 모양새가 다름 -> np.array로 데이터 배열 바꿔줘야함 
'''
36   1  0  0
110  0  1  0
78   0  1  0
62   0  1  0
177  0  0  1
'''

'''
import numpy as np     #numpy 불러오기 : 예시데이터의 NaN생성
import pandas as pd   #pandas 불러오기

fruit = pd.DataFrame({'name':['apple', 'banana', 'cherry', 'durian', np.nan],
                      'color':['red', 'yellow', 'red', 'green', np.nan]})   
                      #예시 데이터 생성
'''


import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score #분류=> 결과지표 'accuracy_score' 떠올라야함
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터 
datasets = load_wine()
print(datasets.DESCR) #(150,4)  #pandas : describe()
'''
#class : label값 (3개를 맞춰라)
 - class_0
 - class_1
 - class_2
'''
print(datasets.feature_names)  #pandas : colums()
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'] 


x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)
print(x)
print(y)  #0,1,2로만 순서대로 나옴-> 섞어줘야함(shuffle)
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2]


##########데이터 분리전에 one-hot encoding하기##########################
#y값 (178, ) -> (178,3) 만들어주기/ pandas get_dummies

import pandas as pd
y=pd.get_dummies(y)
print(y.shape) #(178, 3)
print(y[:3])
'''
   0  1  2
0  1  0  0
1  1  0  0
2  1  0  0
'''
y = np.array(y)
print(y[:5])
'''
[[1 0 0]
 [1 0 0]
 [1 0 0]]
'''
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
model.add(Dense(8, activation='relu', input_dim=13))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax')) #3개의 데이터를 뽑으니까 *label의 개수만큼 노드를 잡는다!! 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_split=0.2,
          verbose=1,
          )


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results)  

y_predict = model.predict(x_test)
# print(y_predict.shape) #(360, 10)
y_predict = np.argmax(y_predict, axis=-1)
print(y_predict.shape) #(360,)

# print(y_test.shape) #(360, 10)
y_test = np.argmax(y_test, axis=-1)
print(y_test.shape) #(360, )

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

'''
36   1  0  0
110  0  1  0
78   0  1  0
62   0  1  0
177  0  0  1
panda => error발생 : (데이터종류) 코드 모양새가 다름 
'''
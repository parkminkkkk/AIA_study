#사이킷런 load_wine

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
#y값 (178, ) -> (178,3) 만들어주기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape) #(178, 3)
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
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


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

model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2,
          verbose=1,
          callbacks=[es]
          )


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results)

y_predict = model.predict(x_test)
# print(y_predict.shape) #(36, 10)
y_predict = np.argmax(y_predict, axis=1) 
print(y_predict.shape) #(36,)
#axis=1, axis=-1동일 (왜냐하면 one-hot하면 2차원으로 데이터가 나오니까 동일함)

# print(y_test.shape) #(36, 10)
y_test = np.argmax(y_test, axis=1)
print(y_test.shape) #(36, )


from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

'''
1. 
results: [0.6034762859344482, 0.8055555820465088]
accuracy_score: 0.8055555555555556

2. *MinMaxScaler
results: [0.09489498287439346, 0.9444444179534912]
acc:  0.9444444444444444
'''
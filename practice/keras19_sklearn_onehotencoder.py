#판다스에 겟더미, 사이킷런에 원핫인코더 (과제)
'''
from sklearn.preprocessing import OneHotEncoder
oht=OneHotEncoder()
y=y(oht.fit_transform(df[['명목변수']]).toarray())
nominal_df
print(y.shape) #(150, 3)
'''
'''
OneHotEncoder()는 명목변수든 순위변수든 모두 원핫 인코딩을 해준다.

pd.get_dummies()함수는 명목변수만 원핫 인코딩을 해준다.
'''

#다중분류
# output layer, loss만 다름
#1. 마지막 아웃풋 레이어, activation = 'softmax'사용 (바뀌지않음!무조건)
#   y의 라벨값의 개수만큼 노드를 정해준다!(이중분류와의 차이점)
#2. loss = 'categorical_crossentropy'사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score #분류=> 결과지표 'accuracy_score' 떠올라야함
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
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
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


#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax')) #3개의 데이터를 뽑으니까 *label의 개수만큼 노드를 잡는다!! 
# softmax : 3개에 대한 확률값으로 뽑음 (모든 확률의 값은 항상 1.0임) / 비율대로 0-1로 한정시키는 함수
# 그 중 가장 높은 확률의 값을 찾아내고, 이를 예측값이라고한다) 

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

# *ValueError: Shapes (16, 1) and (16, 3) are incompatible 
# y의 라벨값의 개수 = y의 클래스의 개수
# (150,)인데 노드3으로 잡으면 (150,3)으로 출력하기때문에 오류가 발생 (..y값1개인데 이 안에 클래스3개 들어가있음..)
# => one-hot Encoding 사용!
#softmax의 전체의 합=1, one-hot encoding의 전체의 합 =1 


#[과제] accuracy_score를 사용해서 스코어를 빼세요.
#y_predict값(소수점나오니까..)=> 0,1,2 로 바꿔줘야함 (np에 있음)


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

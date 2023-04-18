###분류데이터들 모아서 테스트###
# #분류
# 1. iris 
# 2. cancer
# 3. dacon_diabets
# 4. wine
# 5. fetch_covtype
# 6. digits
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits

#1. 데이터 
# x = datasets.data
# y = datasets['target']
# datasets = load_iris()
# datasets = load_breast_cancer()
# datasets = load_wine()
datasets = fetch_covtype()
# datasets = load_digits()
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
# x, y = load_wine(return_X_y=True)
x, y = fetch_covtype(return_X_y=True)
# x, y = load_digits(return_X_y=True)


#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

# model = LinearSVC(C=0.3) #알고리즘 연산이 모델안에 다 포함되어 있음./ #C : 작으면 작을 수록 직선
# model = LogisticRegression()   #Regression: 이름에 들어가지만 분류모델!!**헷갈림주의**
# model = DecisionTreeRegressor()
# model = DecisionTreeClassifier() #트리구조의 모델은 결측치에서 자유롭다
# model = RandomForestRegressor()
#=> 회귀모델(Regressor)은 분류모델(Classifier)로 판단할 수 없다

model1 = RandomForestRegressor()
model2 = DecisionTreeRegressor()
model3 = LogisticRegression()
model4 = LinearSVC()

#DL
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련 
model1.fit(x,y)
model2.fit(x,y)
model3.fit(x,y)
model4.fit(x,y)

#DL
# model.compile(loss = 'sparse_categorical_crossentropy',#원핫인코딩안하면 sparse_categorical_crossentropy사용하면 됨 (대신, 0부터 시작하는지 확인해야함!, 아닐 경우 틀어짐)
#               optimizer='adam',
#               metrics = ['acc'])
# model.fit(x,y, epochs = 100, validation_split=0.2)


#4. 평가, 예측 
###model.score : 분류모델:acc/ 회귀모델:R2
# results= model.score(x,y) #evaluate없음. score
results1= model1.score(x,y)
results2= model2.score(x,y)
results3= model3.score(x,y)
results4= model4.score(x,y)
print("RandomForestRegressor:", results1)
print("DecisionTreeRegressor:", results2)
print("LogisticRegression:", results3)
print("LinearSVC:", results4)

#DL
# results = model.evaluate(x, y)


'''
#아이리스 
#DL
0.17264685034751892
0.946666657924652

#ML
0.9666666666666667
'''







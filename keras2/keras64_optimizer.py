import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=337, shuffle=True
)

#2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련 

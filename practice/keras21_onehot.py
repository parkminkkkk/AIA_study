
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 

#1. 데이터 
datasets = fetch_covtype()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
print('y의 라벨값 :', np.unique(y)) #y의 라벨값 : [1 2 3 4 5 6 7] 
print(y)

##############################################################################
#one hot encoding 

#1. 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y=np.delete(y,0,axis=1)
print(y)
print(y.shape) #(581012, 8)
#y의 라벨값 : [1 2 3 4 5 6 7] 
#keras에서는 label값을 1부터 매겨서, 나중에 shape찍었을때는 (0~7)까지이므로, 8로 찍힘..   / 0행에 0만찍히니까 없애줘야함
#y=np.delete(y,0,axis=1)

#2.
import pandas as pd
y=pd.get_dummies(y)
print(y)
print(y.shape) #(581012, 7) 

y = np.array(y) ##0번째 행 삭제..


#3. 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y)
print(y.shape) #(581012, 7) 
#############################################################################
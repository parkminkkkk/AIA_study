#[과제]one-hot encoding 3가지 방법 차이를 정리
#1. keras의 to_categorical
#2. pandas의 get_dummies
#3. sklearn의 OneHotEncoder 


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
#[one hot encoding]#
#(581012, 54) (581012,) -> (581012,7)  

#1. keras의 to_categorical
import numpy as np
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape) #(581012, 8)       #'7'나와야하는데 '8'이나옴#
#y의 라벨값 : [1 2 3 4 5 6 7] 
#keras에서는 label값을 1부터 매겨서, 나중에 shape찍었을때는 (0~7)까지이므로, 8로 찍힘..   / 0열에 0만찍히니까 없애줘야함
y=np.delete(y,0,axis=1) #'0'번째 '열' '삭제' 


#2. pandas의 get_dummies
import pandas as pd
y=pd.get_dummies(y)
print(y.shape)        #(581012, 7) 
print(type(y))
print(y[:3])
'''
   0  1  2
0  1  0  0
1  1  0  0
2  1  0  0
'''
y = np.array(y)    ###앞의 데이터 '0열'삭제 =정렬 / 데이터 타입을 바꿔줌 np로!
print(type(y))          
print(y[:3])
'''
[[1 0 0]
 [1 0 0]
 [1 0 0]]
'''

#3. sklearn의 OneHotEncoder 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
print(y.shape) #(581012,)
print(type(y))
y = y.reshape(-1,1)
print(type(y))
print(y.shape) #(581012, 1)
y = ohe.fit_transform(y).toarray()
print(y.shape) #(581012, 7) 
##############################################################################


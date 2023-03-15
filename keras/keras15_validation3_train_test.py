from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터 
x = np.array(range(1,17))
y = np.array(range(1,17))

#[실습] 슬라이싱 
#train_test_split로만 잘라보기 
#10:3:3

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.625, random_state=1234)         # train, test(val포함) = 10:6으로 나눔  
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5, random_state=1234)    # test, val을 반으로 나눔 


print(x_train)  
print(x_test)   
print(x_val)
#[11  8  2 10  9  5  6  7  4 16]
#[14 15 13]
#[ 1 12  3]


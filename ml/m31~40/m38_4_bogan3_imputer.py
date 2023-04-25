import pandas as pd
import numpy as np
import sklearn as sk
print(sk.__version__) #1.2.2


data = pd.DataFrame([[2, np.nan, 6, 8, 10], 
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]).transpose()  #transpose : 행과 열 바꾸기
# print(data)
# print(data.shape) #(5, 4)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)


from sklearn.experimental import enable_iterative_imputer #IterativeImputer : 정식버전 아님 / SimpleImputer, KNNImputer : 정식버전
from sklearn.impute import IterativeImputer #interpolation과 비슷(선형회귀선 값과 비슷 )
from sklearn.impute import SimpleImputer #돌리다, 전가하다(결측치에 대한 책임을 돌리다..) => 결측치 대체
from sklearn.impute import KNNImputer    #최근접이웃값 대체 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# imputer = SimpleImputer()  #디폴트 : 평균값 (각 열의 평균값으로 각 열의 Nan값 대체) 
# imputer = SimpleImputer(strategy='mean')        #평균값
# imputer = SimpleImputer(strategy='median')      #중위값 
# imputer = SimpleImputer(strategy='most_frequent') #최빈값 (개수 똑같을 때, 최빈값 중 가장 작은값)
# imputer = SimpleImputer(strategy='constant',fill_value=7777)     #일반값 넣을때 사용가능 #0들어감(디폴트) #fill_value=7777 : 7777들어감 
# imputer = IterativeImputer()       #(선형회귀선 값과 비슷)
imputer = IterativeImputer(estimator=XGBRegressor())  #트리계열 모델을 파라미터로 사용가능 / XGBRegressor, DecisionTreeRegressor 등등.. 
data2 = imputer.fit_transform(data)
print(data2)

#numpy 입력 받고 numpy로 반환 함 
#pd로 불러올 경우, 다시 DataFrame으로 반환해줘야함 

'''
#print(data)
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN

#print(data2)
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
'''

'''
#imputer = SimpleImputer(strategy='most_frequent')
0  12.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
[[12.  2.  2.  4.]
 [ 6.  4.  4.  4.]
 [ 6.  2.  6.  4.]
 [ 8.  8.  8.  8.]
 [10.  2. 10.  4.]]

'''

'''
#imputer = SimpleImputer(strategy='constant',fill_value=7777) 

[[2.000e+00 2.000e+00 2.000e+00 7.777e+03]
 [7.777e+03 4.000e+00 4.000e+00 4.000e+00]
 [6.000e+00 7.777e+03 6.000e+00 7.777e+03]
 [8.000e+00 8.000e+00 8.000e+00 8.000e+00]
 [1.000e+01 7.777e+03 1.000e+01 7.777e+03]]
'''

'''
# imputer = IterativeImputer()  
[[ 2.          2.          2.          2.0000005 ]
 [ 4.00000099  4.          4.          4.        ]
 [ 6.          5.99999928  6.          5.9999996 ]
 [ 8.          8.          8.          8.        ]
 [10.          9.99999872 10.          9.99999874]]
'''

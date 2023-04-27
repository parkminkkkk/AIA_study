#MICE : Multiple Imputation by Chained Equations 

import pandas as pd
import numpy as np
import sklearn as sk
from impyute.imputation.cs import mice

data = pd.DataFrame([[2, np.nan, 6, 8, 10], 
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]).transpose()  #transpose : 행과 열 바꾸기
# print(data)
# print(data.shape) #(5, 4)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

impute_df = mice(data.to_numpy())  #pd=>numpy로 변경 (mice에서는 numpy형태로 넣어주기) 
impute_df = mice(data.values)  #pd=>numpy로 변경 (mice에서는 numpy형태로 넣어주기) 
print(impute_df)

#AttributeError: 'DataFrame' object has no attribute 'as_matrix'   (pd형식일때 error발생)
#=>> df.as_matrix()버전 0.23.0 이후에는 더 이상 사용되지 않습니다. 대신 사용 df.values하십시오. #pd=>numpy로 변경

'''
[[ 2.          2.          2.          2.03398293]
 [ 3.99028709  4.          4.          4.        ]
 [ 6.          5.69429129  6.          5.91056947]
 [ 8.          8.          8.          8.        ]
 [10.          8.77716514 10.          9.58563965]]
'''
#선형방식으로 NaN값 찾아줌.
#즉, interpolation, mice두개 모두 가능
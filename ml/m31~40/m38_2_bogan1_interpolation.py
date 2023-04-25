import numpy as np
import pandas as pd
from datetime import datetime

dates = ['4/25/2023', '4/26/2023', '4/27/2023', '4/28/2023', '4/29/2023', '4/30/2023']

dates = pd.to_datetime(dates)
print(dates)
# DatetimeIndex(['2023-04-25', '2023-04-26', '2023-04-27', '2023-04-28',
#                '2023-04-29'],
#               dtype='datetime64[ns]', freq=None)
print(type(dates))  
#<class 'pandas.core.indexes.datetimes.DatetimeIndex'>

print("======================================")
# ts = pd.Series 
#pd 데이터 1개 있을때 Series : 벡터(1차원), 컬럼(열)하나와 매치가능/ pd 데이터 2개이상 있을때 DataFrame : 매트릭스(2차원,행렬), Series가 여러개 모인 것
ts = pd.Series([2, None, np.nan, 8, 10, np.nan], index=dates) #np.nan , None(파이썬) 둘다 가능 
print(ts)
'''
2023-04-25     2.0
2023-04-26     NaN
2023-04-27     NaN
2023-04-28     8.0
2023-04-29    10.0
2023-04-30     NaN
dtype: float64
'''
print("======================================")
ts = ts.interpolate() 
print(ts)
'''
2023-04-25     2.0
2023-04-26     4.0
2023-04-27     6.0
2023-04-28     8.0
2023-04-29    10.0
2023-04-30    10.0   
dtype: float64
'''
#맨 끝의 값은 이전 값과 그을 포인트가 없어서 앞의 값 ffill로 채워짐




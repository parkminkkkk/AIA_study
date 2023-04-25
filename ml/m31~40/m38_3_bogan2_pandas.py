import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10], 
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]).transpose()  #transpose : 행과 열 바꾸기
# print(data)
# print(data.shape) #(5, 4)
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
'''
#최종 데이터 형태
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

# #0. 결측치 확인 
# print(data.isnull())  #True : 결측치 있음
# print(data.isnull().sum())
# print(data.info())

# #1. 결측치 삭제 
# print("===============결측치삭제======================")
# print(data['x1'].dropna())  #해당 열에서만 삭제 되어서 의미없음
# print("===============결측치삭제(axis=0)======================")
# print(data.dropna())        #행 위주 삭제 / 디폴트 (nan값 있는 해당 행이 모두 삭제 됨 / 3번째 행만 남게됨)
# print(data.dropna(axis=0))  #행 위주 삭제 / 디폴트
# print("===============결측치삭제(axis=1)======================")
# print(data.dropna(axis=1))  #열 위주 삭제 (x3열만 남음)

# #2-1. 특정값 대체 
# print("===============결측치처리 mean()======================")
# means = data.mean()
# print('평균:', means)
# data2 = data.fillna(means)
# print(data2)
# print("===============결측치처리 median()======================")
# median = data.median()
# print('중앙값:', median)
# data3 = data.fillna(median)
# print(data3)
# print("===============결측치처리 fillna()======================")
# data4 = data.fillna(0)
# print(data4)
# print("===============결측치처리 ffill/bfill()======================")
# data5 = data.fillna(method='ffill')       #주의**맨 처음 값은 가져올 값이 없어져 채워지지 않음**Nan
# print(data5)
# data6 = data.fillna(method='bfill')       #주의**맨 마지막 값은 가져올 값이 없어져 채워지지 않음**Nan
# print(data6)
# print("===============결측치처리 임의값()======================")
# data7 = data.fillna(value=7777)          
# print(data7)

#############특정 컬럼만!!!########################################
#yys
#1. x1컬럼에 평균값 넣기 
# mean = data['x1'].mean()
# print(mean)  #6.5
# data['x1'] = data['x1'].fillna(mean)
data['x1'] = data['x1'].fillna(data['x1'].mean()) #한 줄 코드 
# data['x1'] = data['x1'].fillna(data.mean()) #XXX안됨XXX  
print(data) 
#2. x2컬럼에 중위값 넣기
medians = data['x2'].median()
print(medians) #4.0
data['x2'] = data['x2'].fillna(medians)
print(data)
#3. x4컬럼에 ffill 한 후 제일 위에 남은 행에 7777로 채우기 
data['x4'] = data['x4'].fillna(method='ffill').fillna(value=7777)
# data['x4'] = data['x4'].ffill().fillna(value=7777)
print(data)

'''
#방법2
# df.loc[df['A'] != df['A'], 'A'] = df['A'].mean()

#1. x1컬럼에 평균값 넣기 
data.loc[data['x1'] != data['x1'],'x1'] = data['x1'].mean()
print(data)

#2. x2컬럼에 중위값 넣기
data.loc[data['x2'] != data['x2'],'x2'] = data['x2'].median()
print(data)

#3. x4컬럼에 ffill 한 후 제일 위에 남은 행에 7777로 채우기 
data.loc[data['x4'] != data['x4'],'x4'] = data['x4'].ffill()     
# data.loc[data['x4'] != data['x4'],'x4'] = data['x4'].fillna(method='ffill)     
data.loc[data['x4'] != data['x4'],'x4'] = data['x4'].fillna(7777)
print(data)
'''
#############특정 컬럼만!!!########################################




'''
#print(data.isnull())
      x1     x2     x3     x4
0  False  False  False   True
1   True  False  False  False
2  False   True  False   True
3  False  False  False  False
4  False   True  False   True

#print(data.isnull().sum())
x1    1
x2    2
x3    0
x4    3
dtype: int64

#print(data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   x1      4 non-null      float64
 1   x2      3 non-null      float64
 2   x3      5 non-null      float64
 3   x4      2 non-null      float64
dtypes: float64(4)
memory usage: 288.0 bytes
None
===============결측치삭제======================
#print(data['x1'].dropna()) 

0     2.0
2     6.0
3     8.0
4    10.0
Name: x1, dtype: float64
===============결측치삭제(axis=0)======================
    x1   x2   x3   x4
3  8.0  8.0  8.0  8.0
    x1   x2   x3   x4
3  8.0  8.0  8.0  8.0
===============결측치삭제(axis=1)======================
     x3
0   2.0
1   4.0
2   6.0
3   8.0
4  10.0
===============결측치처리 mean()======================
평균: x1    6.500000
x2    4.666667
x3    6.000000
x4    6.000000
dtype: float64
     x1        x2    x3   x4
0   2.0  2.000000   2.0  6.0
1   6.5  4.000000   4.0  4.0
2   6.0  4.666667   6.0  6.0
3   8.0  8.000000   8.0  8.0
4  10.0  4.666667  10.0  6.0
===============결측치처리 median()======================
중앙값: x1    7.0
x2    4.0
x3    6.0
x4    6.0
dtype: float64
     x1   x2    x3   x4
0   2.0  2.0   2.0  6.0
1   7.0  4.0   4.0  4.0
2   6.0  4.0   6.0  6.0
3   8.0  8.0   8.0  8.0
4  10.0  4.0  10.0  6.0
===============결측치처리 fillna()======================
     x1   x2    x3   x4
0   2.0  2.0   2.0  0.0
1   0.0  4.0   4.0  4.0
2   6.0  0.0   6.0  0.0
3   8.0  8.0   8.0  8.0
4  10.0  0.0  10.0  0.0
===============결측치처리 ffill/bfill()======================
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN    #주의**맨 처음 값은 가져올 값이 없어져 채워지지 않음**Nan
1   2.0  4.0   4.0  4.0
2   6.0  4.0   6.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0
     x1   x2    x3   x4
0   2.0  2.0   2.0  4.0
1   6.0  4.0   4.0  4.0
2   6.0  8.0   6.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN   #주의**맨 마지막 값은 가져올 값이 없어져 채워지지 않음**Nan
===============결측치처리 임의값()======================
       x1      x2    x3      x4
0     2.0     2.0   2.0  7777.0
1  7777.0     4.0   4.0     4.0
2     6.0  7777.0   6.0  7777.0
3     8.0     8.0   8.0     8.0
4    10.0  7777.0  10.0  7777.0
=======================================================
'''

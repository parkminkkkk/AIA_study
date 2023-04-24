import numpy as np
import pandas as pd

df = pd.DataFrame({'A' : [1,2,3,4,5],
                   'B' : [10,20,30,40,50],
                   'C' : [5,4,3,2,1]})
print(df) 
#A와 B의 가중치 관계 1 / A와 B는 가중치 같고, bias만 다름
#B와 C의 가중치 관계 -1 / A와 C의 가중치 관계 -1
#A,A/B,B/C,C 가중치 관계 1

correlations  = df.corr()
print(correlations)   #단순 선형회귀이므로 100%신뢰하면 안됨(성능좋지않은 linear모델 돌린 것이므로, 참고용으로!!)
'''
     A    B    C
A  1.0  1.0 -1.0
B  1.0  1.0 -1.0
C -1.0 -1.0  1.0
'''


import numpy as np

aaa = np.array([-10,2,3,4,5,6,700,
                8,9,10,11,12,50])
# print(aaa.shape) #(13,)
aaa = aaa.reshape(-1,1)  #원래 벡터형태로 적용가능했음
# print(aaa) #(13,1)
'''
#그러나  EllipticEnvelope는 2차원형태를 요구함
ValueError: Expected 2D array, got 1D array instead: 
array=[-10   2   3   4   5   6 700   8   9  10  11  12  50].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample
'''


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2,) #contamination : 전체의 몇프로를 이상치로 할 것인가 (.1 = 10%)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
'''
#contamination=.1
[ 1  1  1  1  1  1 -1  1  1  1  1  1 -1] -1 이상치의 위치 
#contamination=.2
[-1  1  1  1  1  1 -1  1  1  1  1  1 -1]
#contamination=.3
[-1 -1  1  1  1  1 -1  1  1  1  1 -1 -1]
'''

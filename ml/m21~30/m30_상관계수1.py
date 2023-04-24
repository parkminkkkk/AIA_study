import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터 
datasets = load_iris()
print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names) #columns 컬럼명 명시(처음에는 numpy형태로 컬럼없이 들어가있으므로, 컬럼명 추가해주기)
print(df)

df['Target(Y)'] = y     #Target(Y)컬럼을 만들어서 y를 넣어준다. 
print(df)  
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
0                  5.1               3.5                1.4               0.2          0
1                  4.9               3.0                1.4               0.2          0
2                  4.7               3.2                1.3               0.2          0
..                 ...               ...                ...               ...        ...
148                6.2               3.4                5.4               2.3          2
149                5.9               3.0                5.1               1.8          2

[150 rows x 5 columns]
'''
print("==============상관계수 히트 맵 두둥===================")
print(df.corr()) #corr -> 상관성 / #feature_infortance랑 두개 함께 사용하면 좋음
'''
==============상관계수 히트 맵 두둥===================
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000
'''
#상관관계가 높은 Y변수는 좋음!!
#Y와의 상관관계가 높은 것이 좋음  => #'(-)관계인 것은 삭제해도 좋다' 판단가능 [ex)-0.426658]
#상관관계가 높은 X변수:[ex)0.962865] => '두개 모두 상관관계가 높을때는 두개 다 필요 없을 수 있다'고 판단가능/ 두개 중에 하나 삭제 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()
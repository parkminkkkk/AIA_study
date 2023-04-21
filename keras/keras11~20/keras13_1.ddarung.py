#데이콘 따릉이 문제풀이(대회) *중요* 많이 써먹음!! 머리에 다 넣기!!
#train.csv 파일 :  x,y로 구분해서 훈련시키고 
#test.csv 파일 : 그대로 predict파일에 집어넣음 (count값 없는 파일이므로..)-> count값 구함
#submission.csv 파일 : predict값(count)구한 것을 넣어서 파일 제출!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
 #r2스코어, mse 궁금함 -> 불러옴/ mse지표에 루트씌워서 rmse지표 만들어주기 
import pandas as pd 
 #깔끔하게 데이터화 됨(csv 데이터 가져올때 좋음) *실무에서 엄청 씀 pandas*

#1. 데이터 
path = './_data/ddarung/' 
path_save = './_save/ddarung/' 
 #'.'= 현재폴더(study)  '/'=하단, _data하단의 ddarung데이터 
 # train_csv = pd.read_csv('./_data/ddarung/train.csv')  #원래 이렇게 써야함/ 자주쓰니까 path로 명명

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #0번째는 인덱스 칼럼이야  
print(train_csv) #확인
print(train_csv.shape) # (1459, 10)
# (1459, 11) #id = index , 색인, 아이디 : 번호 매겨놓은것! 데이터 아님!(데이터 불러올때 넣지 않겠다고 명명해야함)
# (1459, 10) 되어야함! (index빼니까) / index_col=0 명시 
# *컬럼명(헤더)와 index는 따로 연산하지 않는다*


test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0) 
print(test_csv)
print(test_csv.shape) #(715, 9) : count 없어서 (10->9)


###확인절차###==============================================================================
# print(train_csv.columns)
'''
#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object')
'''
# print(train_csv.info())
'''
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64   #1459행, 1459개 데이터 
 1   hour_bef_temperature    1457 non-null   float64 #2개 데이터 결측치 
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64 #100여개 데이터 결측치 
 9   count                   1459 non-null   float64
'''
# print(train_csv.describe())
'''
#count  1459.000000           1457.000000             1457.000000         1450.000000        1457.000000          1457.000000     1383.000000    1369.000000     1342.000000  1459.000000
#평균     mean     11.493489             16.717433                0.031572            2.479034          52.231297          1405.216884        0.039149      57.168736       30.327124   108.563400
#표준편차 std       6.922790              5.239150                0.174917            1.378265          20.370387           583.131708        0.019509      31.771019       14.713252    82.631733
#최소값   min       0.000000              3.100000                0.000000            0.000000           7.000000            78.000000        0.003000       9.000000        8.000000     1.000000
#25%값    25%       5.500000             12.800000                0.000000            1.400000          36.000000           879.000000        0.025500      36.000000       20.000000    37.000000
#중위값  50%      11.000000             16.600000                0.000000            2.300000          51.000000          1577.000000        0.039000      51.000000       26.000000    96.000000
#75%값   75%      17.500000             20.100000                0.000000            3.400000          69.000000          1994.000000        0.052000      69.000000       37.000000   150.000000
#최대값  max      23.000000             30.000000                1.000000            8.000000          99.000000          2000.000000        0.125000     269.000000       90.000000   431.000000
'''
# print(train_csv.describe())
'''
*중요*
<class 'pandas.core.frame.DataFrame'>
'''
#=========================================================================================


############# <결측치 처리 먼저> ##########################################################
'''
*선 결측치 후 데이터분리*
-loss=nan나오는 문제 해결하기 위함
-먼저 결측치 처리하는 이유 : 통 데이터일때 결측치 처리한다. 왜냐하면 나중에 할 경우 없어질 수도 있으니까..
 ex) x의 결측치 1000, y의 결측치 1000이라고 할때, 
     x와 y를 먼저 분리한 후, 결측치를 제거한다 한다면 => x는 900개 y는 1000개 되면 순서가 달라지므로 
     선 결측치를 해서 순서를 정렬한 이후에 데이터를 분리한다. 
'''
#1. 결측치 처리 = 제거 
# print(train_csv.isnull()) : true/False
print(train_csv.isnull().sum()) #isnull이 True인것의 합계 : 각 컬럼별로 결측치 몇개인지 알수 있음
train_csv = train_csv.dropna()   ### *결측치제거* 결측치 삭제하겠다 (함수형태) -> train_csv로 명명
print(train_csv.isnull().sum()) #결측치 제거 후 확인용 
print(train_csv.info())
print(train_csv.shape)       #(1328, 10)

##########################################################################################


############# <train_csv데이터에서 x와 y를 분리> ############################################
#y데이터 분리 *가장 중요(외우기)*
#x : count컬럼 뺀 전부 / y : count컬럼
#train.csv 파일 :  x,y로 구분해서 훈련시키고 
#test.csv 파일 : 그대로 predict파일에 집어넣음 (count값 없는 파일이므로..)-> count값 구함
#submission.csv 파일 : predict값(count)구한 것을 넣어서 파일 제출!

x = train_csv.drop(['count'], axis=1) 
print(x)
#1. train_set 
#2. axis=1 : 열(컬럼)  / 컬럼(열)을 drop하려면 axis=1로 설정, index(행)를 drop하려면 axis=0으로 설정 
#3. [] : 2개이상은 list, 여러개로 다른 것도 할 수 있다
#변수지정 : x = train_csv에서 count를 drop시킨것

y = train_csv['count']  
print(y)

###########################################################################################

x_train, x_test, y_train, y_test = train_test_split(
      x, y, shuffle=True, train_size=0.8, random_state=650874
      )
                                                         #결측치 제거 후 (train_size=0.7,random_state=777일때)
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9) -> (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021, ) (438, )    -> (929, ) (399, )

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=3000, batch_size=32,
          verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

'''
이대로 돌리면 nan발생
#loss : nan = 결측치가 너무 많다(데이터없음). 공란에 곱하기 해도 공란이므로.. 
 -1) 결측치처리 : 공란을 0으로 채운다 / 결측치 처리 우선-> 이후 x,y데이터 분리 
'''

'''
*loss값 줄이기 
1. loss :  6179.9736328125
- train_size=0.7, random_state=777, Dense(1,16,8,4,1), mse, epochs=10, batch_size=32, verbose=1
2. loss :  2575.637451171875
- train_size=0.8, random_state=777, Dense(32,16,8,4,1), mse, epochs=1000, batch_size=32, verbose=1
3. loss :  2627.971923828125
- train_size=0.8, random_state=1234, Dense(32,16,8,4,1), mse, epochs=10000, batch_size=32, verbose=1
 
'''


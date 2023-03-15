#실무/ 실제 데이터 대입 *통으로 외우기*
#따릉이문제 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #mse
import pandas as pd

#1. 데이터 
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
#  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count
print(train_csv.shape)  
#(1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
#  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5
print(test_csv.shape) 
#(715, 9) *count 제외된 데이터 

 #print : 에러 발생시 내가 프레임을 잘 적용시킨것이 맞는지 확인하기 위해 확인용! -> 이후 결과도 적어두기
print(train_csv.columns)    #칼럼 확인가능 
print(train_csv.info())     #결측치 확인 가능
print(train_csv.describe()) #각 칼럼별로 count(데이터개수), mean(평균), std(표준편차), min(최소값), 25%,50%,75%, max(최대값)
print(type(train_csv))      #type : pandas 위에 pandad가 잘 적용이 되었구나를 확인

 #결측치 처리 먼저 *선 결측치 후 데이터분리*
print(train_csv.isnull().sum()) # 각 컬럼별로 결측치 개수 알수 있음 
'''
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
'''
train_csv = train_csv.dropna()  #결측치제거 
print(train_csv.isnull().sum()) #결측치 0
'''
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0
'''
print(train_csv.info())
print(train_csv.shape)          #(1328, 10)

 #데이터 분리(train_set) *가장 중요*
x = train_csv.drop(['count'], axis=1)  
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=34553
    )                               
print(x_train.shape, x_test.shape) # (929, 9) (399, 9) * train_size=0.7, random_state=777일 때
print(y_train.shape, y_test.shape) # (929,) (399,)     * train_size=0.7, random_state=777일 때

#2. 모델구성 
model = Sequential()
model.add(Dense(8, input_dim=9))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=3)

#4. 평가, 예측
loss = model. evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
#현재까지 loss는 'mse'로 계산 -> 'rmse'로 변경

#RMSE함수의 정의 
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
#RMSE함수의 실행(사용)
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

#submission.csv 만들기
y_submit = model.predict(test_csv) #위에서 'test_csv'명명 -> test_csv예측값을 y_submit이라 함 
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submit_0306_0819.csv') #파일생성 

'''
*loss값 줄이기 (10개 이상)
1.[0447] loss :  3377.663330078125 / RMSE :  58.11766868637356 /  *[58*58=3377:loss]
 81점 / train_size=0.8, random_state=650874, Dense(32,16,8,4,1), mse, epochs=30, batch_size=32, verbose=1
2. [0515] loss :  2444.00732421875 / RMSE :  49.43689952875558 
 75점 / train_size=0.8, random_state=650874, Dense(32,16,8,4,1), mse, epochs=1000, batch_size=32, verbose=1

3. [0724] loss :  2640.547607421875 / r2스코어 :  0.6007472553545274/ RMSE :  51.38625912310264
74.43점 /train_size=0.8, random_state=777, Dense(16,8,4,1), mse, epochs=10000, batch_size=32, verbose=3
4. loss :  2763.154541015625/ r2스코어 :  0.612239477223445/ RMSE :  52.56571789528728
train_size=0.9, random_state=650874, Dense(16,8,4,1), mse, epochs=2000, batch_size=32, verbose=3
5. [0742] loss :  2412.064697265625/ r2스코어 :  0.6031572169071889/ RMSE :  49.11277445641966
*73.75점 /train_size=0.8, random_state=650874, Dense(8,4,2,1), mse, epochs=3000, batch_size=50, verbose=3
6. [0751.mae]  loss :  36.904273986816406/  r2스코어 :  0.5743379853502371/ RMSE :  50.86483892406423
74.49점 /train_size=0.8, random_state=650874, Dense(8,4,2,1), mae, epochs=3000, batch_size=50, verbose=3
7. [0800.mae]loss :  38.51666259765625 / r2스코어 :  0.5583704085609302/ RMSE :  55.08499324888575
74.93점 /train_size=0.9, random_state=4, Dense(8,4,2,1), mae, epochs=3000, batch_size=50, verbose=3
8. [0804] loss :  2365.754150390625/ r2스코어 :  0.5677189406127994/ RMSE :  48.63902056492722
460.17점 /train_size=0.9, random_state=4, Dense(8,4,2,4,1), mse, epochs=3000, batch_size=100, verbose=3
9. [0808] loss :  2304.043212890625/ r2스코어 :  0.6363604008732608/ RMSE :  48.000449521918824
74.42점 /train_size=0.9, random_state=4432, Dense(8,4,2,1), mse, epochs=1000, batch_size=10, verbose=3
10.[0808]2 loss :  3157.8974609375/ r2스코어 :  0.46562312869645317/ RMSE :  56.19517254367024
train_size=0.9, random_state=19315, Dense(8,4,2,1), mse, epochs=3000, batch_size=10, verbose=3
11.[0819] loss :  2533.763916015625/r2스코어 :  0.5651712327634022/RMSE :  50.33650412354253
train_size=0.8, random_state=34553, Dense(8,4,2,1), mse, epochs=1000, batch_size=100, verbose=3

'''
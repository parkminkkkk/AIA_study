'''
**시계열데이터 -날짜 자르기(년,월,일,시,분,초)** 
-주식의 경우 : 년,월,일,시,분,초,토,일,공휴일...

#방법1. 
-(컬럼 소량(pm) 사용/x,y데이터 만들어서 사용)
-y값 명확, PM2.5데이터별로 날짜별로 자르기, x는 2일치(48개) y는 3일치(72개) : 2일치 데이터로 3일치 P.M예측
-train데이터 y값 결측치(imputer) => 이후 여기있는 데이터 사용 (48개 다음엔 72개야, 반복하면서 예측)
-(35000,48)(35000,72) 만들기 / 날짜데이터까지 잘라서 사용할 경우 (35000,4,48)(35000,72) 
-LSTM/ Conv1D모델 사용 
-다차원 xgboost돌릴 수 없음 (머신러닝모델은 3차원 안먹힘 ) => (35000,4*48)(35000,72)해서 shape만 맞춰서 돌릴 수는 있음 

#방법2.
-라벨인코더 => 지역(수치화로 변경)
-연도별로 비교 => 코로나 시점 전/후로 총 미세먼지 양의 차이가 날 수 있음(1,2,3,4)중에서 하나가 커질 수 있음.. -> linear형태
-일시 중요 => 서쪽에서 바람불때(겨울) // 남쪽에서 바람불때(여름)  : 12,1,2월에 미세먼지 더 많을 확률 높음 : 월을 데이터로 사용가능, 그러나 일은 데이터로 사용하기에 명확하지 xx
-시간데이터 => 낮>밤 // 아침<점심 
-연도, 일시, 지역, pm(y) => 일시는 월과 일로 잘라줌 - 컬럼 4개로 늘어남 / 지역데이터는 라벨인코더 
-즉, ㅇㅇㅇㅇ일때 y값이야! 를 맞추는 모델 : 35000번해서 가중치 생성 => 이후, test데이터로 확인 

#AWS : 30개 지점 있음/ 최근접이웃 - 특정 지점이 가까이 있다면 3개중 1개 선택해서 데이터 뒤에 붙이면 됨.. (성능 더 안좋아질 수도 있음..왜냐하면 가중치 가장 높은 것은 PM2.5이므로,,)
#따라서, PM2.5가 가중되는 것... 잘 찾기... 

'''
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

#// \\ / \ 모두 동일함 
#\n : 줄바꿈 , \a : 띄어쓰기 , \t: tab 등 예약어로 인식함(노란색으로 색바뀜) : 따라서 경로에서 하위디렉토리 명시할때에는 \\ 두개로 사용

# Load the data
path = 'D:/study/_data/AIFac_pollution/'   
save_path = './_save/AIFac_pollution/'

# train_data = pd.read_csv(path + 'TRAIN.csv', index_col=0)
# train_awsdata = pd.read_csv(path + 'TRAIN_AWS.csv')
# test_data = pd.read_csv(path + 'TEST_INPUT.csv')
# test_awsdata = pd.read_csv(path + 'TEST_AWS.csv')
# meta_data = pd.read_csv(path + 'META.csv')
# submission = pd.read_csv(path + 'answer_sample.csv')

train_files = glob.glob(path + "TRAIN/*.csv") #이 폴더 안에 있는 모든 데이터를 가져와서 텍스트화 시켜줌
# print(train_files)
test_files = glob.glob(path + "test_input/*.csv") #경로에서는 대소문자 상관x
# print(test_files) #리스트형태 -> for문 

########################################1. 데이터 파일 합치기#################################################
###train files###
dflist = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0, #인덱스 칼럼 없으니까 None, 헤더는 0번째 칼럼
                     encoding='utf-8-sig') 
    dflist.append(df)
# print(dflist)       #[35064 rows x 4 columns] X17개 
# print(len(dflist))  #17 : list형태로 묶여진 것일 뿐 데이터프레임 형태로 된 것은 아님 -> concat 해줘야함
train_data = pd.concat(dflist, axis=0, ignore_index=True) #행단위로 합침 #리스트로 합치면서 index가 동일해졌으므로 무시하고 새로운 index생성해줌
# print(train_data)   #[596088 rows x 4 columns]
#--------------------------------------------------------------------------------------------------------------------------------------------#
###test files###
dflist = []
for filename in test_files:
    df = pd.read_csv(filename, index_col=None, header=0, #인덱스 칼럼 없으니까 None, 헤더는 0번째 칼럼
                     encoding='utf-8-sig') 
    dflist.append(df)
# print(dflist)       #[7728 rows x 4 columns] X17개 
# print(len(dflist))  #17 : list형태로 묶여진 것일 뿐 데이터프레임 형태로 된 것은 아님 -> concat 해줘야함
test_data = pd.concat(dflist, axis=0, ignore_index=True) #행단위로 합침 #리스트로 합치면서 index가 동일해졌으므로 무시하고 새로운 index생성해줌
# print(test_data)   #[131376 rows x 4 columns]

#######################################2. 측정소 위치 라벨인코더 #################################################
le = LabelEncoder()
train_data['location'] = le.fit_transform(train_data['측정소'])   #copy개념으로 새로운 공간(location)에 le만들어줌  (바로 (측정소)해줘도 되긴 함..)
test_data['location'] = le.transform(test_data['측정소'])         #데이터의 위치나 개수에 따라서 바뀔 수 있기때문에, train의 fit한거에 맞춰서 test transform해줘야함**
# print(train_data) #[596088 rows x 5 columns]
# print(test_data)  #[131376 rows x 5 columns]
train_data = train_data.drop(['측정소'], axis=1)
test_data = test_data.drop(['측정소'], axis=1)
# print(train_data) #[596088 rows x 4 columns]
# print(test_data)  #[131376 rows x 4 columns]

#######################################3. 일시-> 년/월/일/시간 분리 #################################################
#12-31 11:30 -> 12(월)와 11(시)추출 
# print(train_data.info())  #['일시'] : object = str형태

train_data['month'] = train_data['일시'].str[:2]   #2번째까지 (0~1번째)
# print(train_data['month'])
train_data['hour'] = train_data['일시'].str[6:8]   #6번째~8번까지 (-, 띄어쓰기도 str에 포함되므로..)
# print(train_data['hour'])
train_data = train_data.drop(['일시'], axis=1)
# print(train_data)  #[596088 rows x 5 columns] =>[연도  PM2.5  location month hour ] 

test_data['month'] = test_data['일시'].str[:2]   #2번째까지 (0~1번째)
test_data['hour'] = test_data['일시'].str[6:8]   #6번째~8번까지 (-, 띄어쓰기도 str에 포함되므로..)
test_data = test_data.drop(['일시'], axis=1) 
# print(test_data)     #[131376 rows x 5 columns]

### str -> int 변경###---------------------------------------------------------------------
# print(train_data.info()) #object으로 month, hour 생성되었음
# train_data['month'] = pd.to_numeric(train_data['month'])    #pd.to_numeric :str -> 수치형 데이터로 변환
# train_data['month'] = train_data['month'].astype('int32')   #두가지 방법 모두 가능 
train_data['month'] = pd.to_numeric(train_data['month']).astype('int8')     #메모리 줄여주기 위해 ('int8')로 바꿔주기/ 연산량 줄여줄 수 있음
train_data['hour'] = pd.to_numeric(train_data['hour']).astype('int8') 
# print(train_data.info()) 

test_data['month'] = pd.to_numeric(test_data['month']).astype('int8')     #메모리 줄여주기 위해 ('int8')로 바꿔주기/ 연산량 줄여줄 수 있음
test_data['hour'] = pd.to_numeric(test_data['hour']).astype('int8') 
# print(test_data.info()) 


#############################################4. 결측치 ###########################################################
# print(train_data.info())
'''
 #   Column    Non-Null Count   Dtype
---  ------    --------------   -----
 0   연도        596088 non-null  int64
 1   PM2.5     580546 non-null  float64    ###15542개의 결측치 존재###
 2   location  596088 non-null  int32
 3   month     596088 non-null  int8
 4   hour      596088 non-null  int8
dtypes: float64(1), int32(1), int64(1), int8(2)
'''
#각 장소별로 결측치 1000개이상 존재함(덩어리진 곳이 있음) -> interpolate해도 정확하지 않음
#방법1.=>>> model돌려서 예측하는 것이 더 정확함 
#방법2.=>>> drop

###PM2.5 15542개의 결측치 존재###
#전체 596088 -> 580546줄임
train_data = train_data.dropna()
# print(train_data.info())
'''
 #   Column    Non-Null Count   Dtype
---  ------    --------------   -----
 0   연도        580546 non-null  int64
 1   PM2.5     580546 non-null  float64
 2   location  580546 non-null  int32
 3   month     580546 non-null  int8
 4   hour      580546 non-null  int8
dtypes: float64(1), int32(1), int64(1), int8(2)
'''

#############################################5. 파생피처(중요) ###########################################################
#주말, 공휴일 등 만들 수 있음 (여러가지 조합으로 생성하는 파생피처) : 피처엔지니어링 작업에서 굉장히 중요함
#계절(봄/여름/가을/겨울) 시즌을 만들어서 피처 하나 만들어 줄 수 있음. (여름<겨울 : 미세먼지 더 많으므로)

#########################################################################################################################
#########################################################################################################################
#모델 훈련방식 : dense형태로 훈련시켜줌(모델 xgboost사용) / x데이터에 대해서 -> y데이터야 훈련시키고 이후 test를 통해 평가,예측 

#1. 데이터
y = train_data['PM2.5']
x = train_data.drop(['PM2.5'], axis=1)

# print("x데이터", x, '\n', "y데이터", y) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=640874, shuffle=True
) 

#[데이터 전처리]
#scale : 트레인,테스트 스플릿 한 이후에 적용/ 트리계열 모델에서는 이상치,결측치 유연하므로 안해줘도 되지만 해서 좋아질 수도 있다!
#라벨링한 데이터 scale을 한다/안한다? : 굳이 할 필요는 없다. (원핫:데이터 컬럼수 늘어남-> 다시 차원축소)
#->(카테고리형 데이터일 경우, 원핫해준다-> 한개 컬럼 원핫해주면(17개+5개 늘어남)=> 지역데이터로 결과값이 좌지우지 될 수 있음 -> 다시 축소해주기(PCA, LDA..))
## 월, 시간데이터 또한, 원핫(12개로 늘어남) / (왜냐하면, 1월이랑 12월이 12배가 차이나는 것이 아니므로..)
## 월, 시간데이터 : 주기함수에다 넣어서 수정(sin,cos함수...)
#한쪽으로 치우친 데이터 : log변환.. 

parameters = {'n_estimators' : 20000,
              'learning_rate' : 0.3,
              'max_depth': 6,
              'gamma': 0,
              'min_child_weight': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state' : 640,
              'n_jobs' : -1
              }


#2. 모델구성
model = XGBRegressor()

#3. 컴파일, 훈련 
model.set_params(**parameters,                   #컴파일과 비슷하다고 생각
                 eval_metric = 'mae', 
                 early_stopping_rounds = 200,
                 ) 

start = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 1
) 
end = time.time()
print("걸린시간:", round(end-start, 3),"초")
#4. 평가, 예측 

y_predict = model.predict(x_test)

results = model.score(x_test, y_test)
print("model.score:", results)
r2 = r2_score(y_test, y_predict)
print("r2.score:", r2)
mae = mean_absolute_error(y_test, y_predict)
print("mae.score:", mae)


#test_data의 결측치만 뽑아서 predict  => 나오는 값을 submit파일에 넣어서 제출 
# ttest_data = test_data[test_data['PM2.5'].isna().any(axis=1)].drop(['PM2.5'], axis=1)
ttest_data = test_data[test_data['PM2.5'].isnull()].drop(['PM2.5'], axis=1)
# print(ttest_data)

y_submit = model.predict(ttest_data)
submission = pd.read_csv(path + 'answer_sample.csv', index_col=0)
submission['PM2.5'] = y_submit

path_save = './_save/AIFac_pollution/'
submission.to_csv(path_save + 'AIF_1.csv') # 파일생성

# print(submission['PM2.5'])



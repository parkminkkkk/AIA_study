import pandas as pd
import numpy as np
import random
import os
# import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.metrics import f1_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(777)  # Seed 고정
path = 'd:/study/_data/dacon_crime/'
save_path ='./_save/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# x_train = train.drop(['ID', 'TARGET'], axis = 1) 
# y_train = train['TARGET']
# x_test = test.drop('ID', axis = 1)

# train.head()
# test.head()

train_df = train.drop('ID', axis = 1).copy()
test_df = test.drop('ID', axis = 1).copy()


'''
##### Data Preprocessing #################
월 : 사건 발생월 (CATEGORICAL) + 계절변수(CATEGORICAL) 추가 가능
요일 : 월요일 ~ 일요일 (CATEGORICAL) + 주말여부 (BINARY) 추가 가능
시간 : 사건 발생 시각 (CATEGORICAL) -> 구체적으로 세분화된 시각보다는 아침, 점심, 저녁, 밤, 새벽으로 구분해서 진행 [0-5시는 새벽, 5-9시는 아침, 9-17시는 낮, 17-21시는 저녁, 21-24시는 밤]
소관경찰서 : 사건 발생 구역의 담당 경찰서 (CATEGORCAL) - 수치로 비식별화
소관지역 : 사건 발생 구역 (CATEGORCAL) - 수치로 비식별화
사건발생거리 : 가장 가까운 경찰서에서 사건 현장까지의 거리 (Numeric) - Standard Scaling 적용
강수량(mm) (Numeric)
강설량(mm) (Numeric)
적설량(cm) -> 안씀
풍향 : 범죄발생지에서 바람이 부는 방향(최대 360도) (CATEGORICAL) -> 8방위 Or 4방위로 구분하여 진행.
안개 : 가시거리가 1km 미만인 경우 (Binary)
짙은안개 : 가시거리가 200m 미만인 경우 (Binary)
번개 (Binary)
진눈깨비 (Binary)
서리 (Binary)
연기/연무 : 먼지, 연기가 하늘을 가리는 현상 (Binary)
눈날림 (Binary)
범죄발생지 : 범죄가 발생한 장소 (CATEGORY)
'''

### 월 변수를 활용해 계절변수 추가
import numpy as np

# 봄, 여름, 가을, 겨울 (1,2,3,4)
season = np.zeros(len(train_df))
season[train_df.query('월 >= 3 and 시간 <= 5').index] = 1 
season[train_df.query('월 >= 6 and 시간 <= 8').index] = 2 
season[train_df.query('월 >= 9 and 시간 <= 11').index] = 3
season[train_df.query('월 <= 2 or 시간 == 12').index] = 4

train_df['계절'] = season.astype(int)

### 요일 구분 + 주말여부 추가
from sklearn.preprocessing import LabelEncoder
le_week = LabelEncoder()
le_week.fit(train_df['요일'].astype('category'))

train_df['요일'] = le_week.transform(train_df['요일'].astype('category'))
train_df['주말여부'] = train_df['요일'].isin(['토요일','일요일']).astype('int')

### 시간 구분 - 24시각 기준이 아님? 12시각 기준 (원래 오전, 오후 구분이 있어야...?)
### https://kosis.kr/statHtml/statHtml.do?orgId=132&tblId=DT_13204_3101 (범죄 발생 시간 통계에는 3시간씩 구분하여 나타냈음.)
### 3시간씩 구분
import numpy as np

time_cate = np.zeros(len(train_df))
time_cate[train_df.query('시간 <= 3').index] = 1 
time_cate[train_df.query('시간 > 3 and 시간 <= 6').index] = 2 
time_cate[train_df.query('시간 > 6 and 시간 <= 9').index] = 3
time_cate[train_df.query('시간 > 9').index] = 4

### 카테고리형으로 변환한다.
train_df['시간구분'] = time_cate.astype(int)
train_df['시간구분'] = train_df['시간구분'].astype('category') 

### 소관경찰서 Category형으로 구분
train_df['소관경찰서'] = train_df['소관경찰서'].astype('category') 

### 소관지역 Category형으로 구분
train_df['소관지역'] = train_df['소관지역'].astype('category') 

### 사건발생거리 - 스케일링 하지 않고, 그대로 사용
'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_df[['사건발생거리']])
train_df['사건발생거리'] = scaler.transform(train_df[['사건발생거리']]) 
'''

### 강수량 Binary 화
### train_df['강수량(mm)'] = (train_df['강수량(mm)'] > 0).astype('int')
### 강설량 Binary 화 & 적설량 Drop
### train_df['강설량(mm)'] = (train_df['강설량(mm)'] > 0).astype('int')
train_df = train_df.drop(['적설량(cm)'], axis = 1)

### 풍향 구분
import numpy as np

cardinal_directions = np.zeros(len(train_df))
cardinal_directions[train_df.query('풍향 <= 45').index] = 1 # 북 ~ 북동
cardinal_directions[train_df.query('풍향 > 45 and 풍향 <= 90').index] = 2 # 북동 ~ 동
cardinal_directions[train_df.query('풍향 > 90 and 풍향 <= 135').index] = 3 # 동 ~ 남동
cardinal_directions[train_df.query('풍향 > 135 and 풍향 <= 180').index] = 4 # 남동 ~ 남
cardinal_directions[train_df.query('풍향 > 180 and 풍향 <= 225').index] = 5 # 남 ~ 남서
cardinal_directions[train_df.query('풍향 > 225 and 풍향 <= 270').index] = 6 # 남서 ~ 서
cardinal_directions[train_df.query('풍향 > 270 and 풍향 <= 315').index] = 7 # 서 ~ 북서
cardinal_directions[train_df.query('풍향 > 315 and 풍향 <= 360').index] = 8 # 북서 ~ 북

# 카테고리형으로 변환한다.
train_df['풍향'] = cardinal_directions.astype(int)
train_df['풍향'] = train_df['풍향'].astype('category') 


### 범죄발생지 카테고리형으로 변환
from sklearn.preprocessing import LabelEncoder
le_loc = LabelEncoder()
le_loc.fit(train_df['범죄발생지'].astype('category'))

train_df['범죄발생지'] = le_loc.transform(train_df['범죄발생지'].astype('category'))

# 정수형으로 변환
for k in ['소관경찰서',	'소관지역',	'풍향',	'안개',	'짙은안개',	'번개',	'진눈깨비',	'서리',	'연기/연무',	'눈날림',	'범죄발생지',	'계절',	'주말여부',	'시간구분', 'TARGET']:
  train_df[k] = train_df[k].astype('int')

train_df.head() # 전처리 Finished

###2. 모델 ##############################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['TARGET'], axis = 1), train_df[['TARGET']], test_size = 0.2, random_state = 87932, stratify = train_df['TARGET'])

from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(iterations = 5000, learning_rate = 0.01, loss_function='MultiClass', eval_metric = 'TotalF1:average=Macro', task_type="GPU")
cat_clf.fit(X_train, y_train, eval_set = (X_test, y_test),
            cat_features = ['월',	'요일',	'시간', '소관경찰서',	'소관지역', '풍향',	'안개',	'짙은안개',	'번개',	'진눈깨비',	'서리',	'연기/연무',	'눈날림',	'범죄발생지',	'계절',	'주말여부',	'시간구분'], 
            verbose = 100, use_best_model = True)

# import eli5
# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(lgbm_clf, random_state = 437863).fit(X_test, y_test)
# eli5.show_weights(perm, feature_names = X_test.columns.tolist())

from sklearn.metrics import f1_score
y_pred = cat_clf.predict(X_test)
f1_score(y_test, y_pred, average='macro')


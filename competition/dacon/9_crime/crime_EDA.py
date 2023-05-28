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

x_train = train.drop(['ID', 'TARGET'], axis = 1) 
y_train = train['TARGET']
x_test = test.drop('ID', axis = 1)

train.head()
test.head()

train_df = train.drop('ID', axis = 1).copy()
test_df = test.drop('ID', axis = 1).copy()

#1. Simple EDA
import matplotlib.pyplot as plt

plt.figure(figsize = (8,7), facecolor = 'w')
plt.pie(x = train_df['TARGET'].value_counts(), autopct = '%.2f%%',
       labels = ['Robbery', 'Injury', 'Theft'], shadow = True, explode = [0.02, 0.02, 0.02])
plt.legend()
plt.title("TARGET Proportion")
plt.show()

#================================#
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (18, 6))

plt.subplot(131)
sns.countplot(x = '요일', data = train_df, order = train_df['요일'].value_counts().index, hue = 'TARGET', palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.subplot(132)
sns.countplot(x = '요일', data = train_df, order = train_df['요일'].value_counts().index, palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.subplot(133)
train_df['주말여부'] = train_df['요일'].isin(['토요일','일요일']).astype('int')
sns.countplot(x = '주말여부', data = train_df, order = train_df['주말여부'].value_counts().index, hue = 'TARGET', palette = 'Reds_r')
plt.xticks(rotation = 90)

plt.show()

#================================#
### 시각대 별 구분

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (12, 6))

plt.subplot(121)
sns.countplot(x = '시간', data = train_df, order = train_df['시간'].value_counts().index, hue = 'TARGET', palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.subplot(122)
sns.countplot(x = '시간', data = train_df, order = train_df['시간'].value_counts().index, palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.show()
#================================#
### 사건 발생 거리 Hist.
### 큰 이상 값은 존재하지 않는 것으로 보임 Standard Scaling으로 스케일만 조정 예정
import matplotlib.pyplot as plt

plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.title('Distance from incident Hist.')
plt.xticks(rotation = 90)

sns.histplot(x = '사건발생거리', data = train_df, kde = True, stat = 'density')
plt.xlabel('Distance')

plt.subplot(122)
plt.title('Distance from incident Hist.')
plt.xticks(rotation = 90)

sns.histplot(x = '사건발생거리', data = train_df, hue = 'TARGET', kde = True, stat = 'density')
plt.xlabel('Distance')

plt.show()
#================================#
#범죄발생지 별 Target
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (12, 6))

plt.subplot(121)
sns.countplot(x = '범죄발생지', data = train_df, order = train_df['범죄발생지'].value_counts().index, hue = 'TARGET', palette = 'Reds_r')
plt.xticks(rotation = 90)

plt.subplot(122)
sns.countplot(x = '범죄발생지', data = train_df, order = train_df['범죄발생지'].value_counts().index, palette = 'Reds_r')
plt.xticks(rotation = 90)

plt.show()

#================================#
# 소관경찰서별 범죄발생 빈도 확인

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (20, 8))

plt.subplot(121)
sns.countplot(x = '소관경찰서', data = train_df, order = train_df['소관경찰서'].value_counts().index[:20], hue = 'TARGET', palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.subplot(122)
sns.countplot(x = '소관경찰서', data = train_df, order = train_df['소관경찰서'].value_counts().index[:20], palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.show()

# 소관지역별 범죄발생빈도 확인

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (20, 8))

plt.subplot(121)
sns.countplot(x = '소관지역', data = train_df, order = train_df['소관지역'].value_counts().index[:15], hue = 'TARGET', palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.subplot(122)
sns.countplot(x = '소관지역', data = train_df, order = train_df['소관지역'].value_counts().index, palette = 'Blues_r')
plt.xticks(rotation = 90)

plt.show()

#==================================#




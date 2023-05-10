import numpy as np
import pandas as pd
import glob # 폴더의 모든 걸 가져와서 텍스트화한다.
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error,r2_score

# / // \ \\ 같다 (\n처럼 예약어가 명시되면 노란색으로 표시될수있으니까 주의해서 쓰자)

path ='./_data/AI_SPARK/'
''' 
-data하단
    TRAIN
    TRAIN_AWS
    TEST_INPUT
    TEST_AWS
    META
    answer.sample.py
'''
save_path = './_save/AI_SPARK/'

# train_data = pd.read_csv(path + 'TRAIN.csv')
# train_aws = pd.read_csv(path + 'TRAIN_AWS.csv')
# test_data = pd.read_csv(path + 'TEST.csv')
# test_aws = pd.read_csv(path + 'TEST_AWS.csv')
# submission = pd.read_csv(path + 'answer_sample.csv')

train_files = glob.glob(path +'TRAIN/*.csv')
# print(train_files) # 리스트 형태
test_input_files = glob.glob(path +'test_input/*.csv')
# print(test_input_files)
train_aws = glob.glob(path + 'train_aws/*.csv')
print(train_aws)

########## train ###########
li=[]
for filename in train_files:
    df = pd.read_csv(filename,index_col=None, header=0,
                     encoding='UTF-8-sig')
    li.append(df) # [35064 rows x 4 columns]
print(len(li)) #17

train_dataset = pd.concat(li,axis=0, ignore_index=True) 
print(train_dataset) #[596088 rows x 4 columns]

########### test #############
li=[]
for filename in test_input_files:
    df = pd.read_csv(filename,index_col=None, header=0,
                     encoding='UTF-8-sig')
    li.append(df) 
print(li)  #[7728 rows x 4 columns]
print(len(li)) #17

test_input_files = pd.concat(li,axis=0, ignore_index=True) 
# print(test_input_files) #[131376 rows x 4 columns]

############# train_aws 이어붙일 파일 이름 리스트에 저장 #####################
import os

file_names = ['공주.csv', '계룡.csv', '논산.csv', '대천항.csv', '대산.csv',
              '태안.csv', '아산.csv', '세천.csv', '성거.csv', '세종전의.csv',
              '세종연서.csv', '세종고운.csv', '예산.csv', '장동.csv', '태안.csv',
              '오월드.csv', '홍북.csv']

# 파일 경로와 파일명을 이용하여 파일 리스트 생성
file_paths = [path+ 'TRAIN_AWS/'+os.path.basename(file) for file in file_names]

# 파일 내용을 담을 빈 데이터프레임 생성
train_aws = pd.DataFrame()

# 선택한 파일들을 이어붙이기
for file_path in file_paths:
    df = pd.read_csv(file_path)
    train_aws = pd.concat([train_aws, df])

# 결과 확인
print(train_aws)

print(train_aws.shape, train_dataset.shape,test_input_files.shape)

############# train_aws 이어붙일 파일 이름 리스트에 저장 #####################

import os

file_names = ['공주.csv', '계룡.csv', '논산.csv', '대천항.csv', '대산.csv',
              '태안.csv', '아산.csv', '세천.csv', '성거.csv', '세종전의.csv',
              '세종연서.csv', '세종고운.csv', '예산.csv', '장동.csv', '태안.csv',
              '오월드.csv', '홍북.csv']

# 파일 경로와 파일명을 이용하여 파일 리스트 생성
file_paths = [path+ 'TEST_AWS/'+os.path.basename(file) for file in file_names]

# 파일 내용을 담을 빈 데이터프레임 생성
test_aws = pd.DataFrame()

# 선택한 파일들을 이어붙이기
for file_path in file_paths:
    df = pd.read_csv(file_path)
    test_aws = pd.concat([test_aws, df])

# 결과 확인
print(test_aws)
print(test_aws.shape)
##############  rename #################

train_names = ['공주.csv', '노은동.csv', '논산.csv', '대천2동.csv', '독곶리.csv',
              '동문동.csv', '모종동.csv', '문창동.csv', '성성동.csv', '신방동.csv',
              '신흥동.csv', '아름동.csv', '예산군.csv', '읍내동.csv', '이원면.csv',
              '정림동.csv', '홍성읍.csv']

# 파일 이름을 train_names 리스트의 순서에 맞게 매칭시켜주는 딕셔너리 생성
name_mapping = {}
for i, name in enumerate(train_names):
    name_mapping[name] = file_names[i]

# AWS 파일 이름을 train_names 리스트의 순서에 맞게 변경
aws_file_names = []
for name in train_names:
    aws_file_names.append(name_mapping[name])
    
print(aws_file_names)

############# 라벨 인코더 ################
le = LabelEncoder()         # 라벨링한 데이터를 스케일러 할수있을까? ->  no!!!!일걸. 원핫해줘 -> 컬럼이 늘어나 -> 과적합될수있어 -> PCA로 줄여 / 스케일러 말고 주기함수..?사용
train_dataset['locate']=le.fit_transform(train_dataset['측정소']) #locate 메모리 공간 만듦 (즉 컬럼이 다섯개 됨)
test_input_files['locate'] = le.transform(test_input_files['측정소'])
print(train_dataset) # [596088 rows x 5 columns]
print(test_input_files) # [131376 rows x 5 columns]

# print(train_aws.info())
# print(test_aws.info())

le2 = LabelEncoder()         # 라벨링한 데이터를 스케일러 할수있을까? ->  no!!!!일걸. 원핫해줘 -> 컬럼이 늘어나 -> 과적합될수있어 -> PCA로 줄여 / 스케일러 말고 주기함수..?사용

train_aws['locate'] = le2.fit_transform(train_aws['지점']) 
# test_aws['locate'] = le.transform(test_aws['지점']) 
print(train_aws)
# print(test_aws)


#측정소 필요없으니까 drop
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_files = test_input_files.drop(['측정소'], axis=1)
print(train_dataset) #[596088 rows x 4 columns]
print(test_input_files) #[131376 rows x 4 columns]
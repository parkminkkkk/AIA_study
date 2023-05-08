import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
train = pd.read_csv('d:/study/_data/AIFac_pollution/train_all.csv')  # PM2.5 파일
train_aws = pd.read_csv('d:/study/_data/AIFac_pollution/train_aws_all.csv')  # AWS 파일

test = pd.read_csv('d:/study/_data/AIFac_pollution/test_all.csv')  # PM2.5 파일
test_aws = pd.read_csv('d:/study/_data/AIFac_pollution/test_aws_all.csv')  # AWS 파일

meta = pd.read_csv('d:/study/_data/AIFac_pollution/meta_all.csv') # meta 파일

submission = pd.read_csv('d:/study/_data/AIFac_pollution/answer_sample.csv')

save_path= './_save/AIFac_pollution/'

print(train.shape)     #(596088, 4)
print(train_aws.shape) #(1051920, 8)
print(test.shape)      #(131376, 4)    #연도,일시,측정소,PM2.5
print(test_aws.shape)  #(231840, 8)    #연도,일시,지점,기온(°C),풍향(deg),풍속(m/s),강수량(mm),습도(%)

# aws의 지점과 train/test의 측정소와 이름을 같게한다.
train_aws = train_aws.rename(columns={"지점": "측정소"})
test_aws = test_aws.rename(columns={"지점": "측정소"})

# train과 train_aws 데이터셋을 지점(station)을 기준으로 merge
train_merged_data = pd.merge(train, train_aws, left_on=["측정소", "일시",'연도'], right_on=["측정소", "일시",'연도'])
print(train_merged_data)   #[70128 rows x 9 columns]
print(train_merged_data.columns)
print(train_merged_data.isnull().sum())
'''
['연도', '일시', '측정소', 'PM2.5', '기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)','습도(%)']
'''
# test와 test_aws와 merge
test_merged_data = pd.merge(test, test_aws, left_on=["측정소", "일시",'연도'], right_on=["측정소", "일시",'연도'])
print(test_merged_data.columns)
'''
['연도', '일시', '측정소', 'PM2.5', '기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)','습도(%)']
'''
# train 데이터와 meta 데이터를 지점 컬럼을 기준으로 merge
meta = meta.rename(columns={"Location": "측정소"})
meta = meta.rename(columns={"Latitude": "위도"})
meta = meta.rename(columns={"Longitude": "경도"})

# Meta 데이터에서 지점 기준으로 위도와 경도 정보 추출
meta_location = meta.groupby('측정소')[['위도', '경도']].mean()
print(train_merged_data)
print(test_merged_data)
print(meta_location)

# train_merged_data와 지점 기준으로 Meta 데이터와 merge 수행
train_merged_data = pd.merge(train_merged_data, meta_location, on='측정소')
test_merged_data = pd.merge(test_merged_data, meta_location, on='측정소')
print(train_merged_data)
print(test_merged_data)




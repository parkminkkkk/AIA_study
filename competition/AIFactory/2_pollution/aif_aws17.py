sego_test_csv=pd.read_csv(path_test_AWS +'세종고운.csv',encoding='utf-8',index_col=False)
seyun_test_csv=pd.read_csv(path_test_AWS +'세종연서.csv',encoding='utf-8',index_col=False)
gyeryong_test_csv=pd.read_csv(path_test_AWS +'계룡.csv',encoding='utf-8',index_col=False)
Oworld_test_csv=pd.read_csv(path_test_AWS +'오월드.csv',encoding='utf-8',index_col=False)
jangdong_test_csv=pd.read_csv(path_test_AWS +'장동.csv',encoding='utf-8',index_col=False)
Oworld_test_csv02=pd.read_csv(path_test_AWS +'오월드.csv',encoding='utf-8',index_col=False)
gongjoo_test_csv=pd.read_csv(path_test_AWS +'공주.csv',encoding='utf-8',index_col=False)

nonsan_test_csv=pd.read_csv(path_test_AWS +'논산.csv',encoding='utf-8',index_col=False)
deacheon_test_csv=pd.read_csv(path_test_AWS +'대천항.csv',encoding='utf-8',index_col=False)
deasan_test_csv=pd.read_csv(path_test_AWS +'대산.csv',encoding='utf-8',index_col=False)
teaan_test_csv=pd.read_csv(path_test_AWS +'태안.csv',encoding='utf-8',index_col=False)
asan_test_csv=pd.read_csv(path_test_AWS +'아산.csv',encoding='utf-8',index_col=False)
sung_test_csv=pd.read_csv(path_test_AWS +'성거.csv',encoding='utf-8',index_col=False)
yesan_test_csv=pd.read_csv(path_test_AWS +'예산.csv',encoding='utf-8',index_col=False)
teaan02_test_csv=pd.read_csv(path_test_AWS +'태안.csv',encoding='utf-8',index_col=False)
aan_test_csv=pd.read_csv(path_test_AWS +'홍북.csv',encoding='utf-8',index_col=False)
sung02_test_csv=pd.read_csv(path_test_AWS +'성거.csv',encoding='utf-8',index_col=False)

test_csv_list = [sego_test_csv,seyun_test_csv,
    gyeryong_test_csv,Oworld_test_csv,
    jangdong_test_csv,Oworld_test_csv02,
    gongjoo_test_csv,nonsan_test_csv,
    deacheon_test_csv,deasan_test_csv,
    teaan_test_csv,asan_test_csv,
    sung_test_csv,yesan_test_csv,
    teaan02_test_csv,aan_test_csv,sung02_test_csv
]

for v in test_csv_list:
    mode = v['지점'].mode()[0]
    v['지점'] = v['지점'].fillna(mode)
print('Done.')

test_aws_dataset = pd.concat([
    sego_test_csv,seyun_test_csv,
    gyeryong_test_csv,Oworld_test_csv,
    jangdong_test_csv,Oworld_test_csv02,
    gongjoo_test_csv,nonsan_test_csv,
    deacheon_test_csv,deasan_test_csv,
    sung_test_csv,yesan_test_csv,
    teaan_test_csv,asan_test_csv,
    teaan02_test_csv,aan_test_csv,sung02_test_csv],axis=0,ignore_index=True
)


sego_train_csv=pd.read_csv(path_train_AWS +'세종고운.csv',encoding='utf-8',index_col=False)
seyun_train_csv=pd.read_csv(path_train_AWS +'세종연서.csv',encoding='utf-8',index_col=False)
gyeryong_train_csv=pd.read_csv(path_train_AWS +'계룡.csv',encoding='utf-8',index_col=False)
Oworld_train_csv=pd.read_csv(path_train_AWS +'오월드.csv',encoding='utf-8',index_col=False)
jangdong_train_csv=pd.read_csv(path_train_AWS +'장동.csv',encoding='utf-8',index_col=False)
Oworld_train_csv02=pd.read_csv(path_train_AWS +'오월드.csv',encoding='utf-8',index_col=False)
gongjoo_train_csv=pd.read_csv(path_train_AWS +'공주.csv',encoding='utf-8',index_col=False)

nonsan_train_csv=pd.read_csv(path_train_AWS +'논산.csv',encoding='utf-8',index_col=False)
deacheon_train_csv=pd.read_csv(path_train_AWS +'대천항.csv',encoding='utf-8',index_col=False)
deasan_train_csv=pd.read_csv(path_train_AWS +'대산.csv',encoding='utf-8',index_col=False)
teaan_train_csv=pd.read_csv(path_train_AWS +'태안.csv',encoding='utf-8',index_col=False)
asan_train_csv=pd.read_csv(path_train_AWS +'아산.csv',encoding='utf-8',index_col=False)
sung_train_csv=pd.read_csv(path_train_AWS +'성거.csv',encoding='utf-8',index_col=False)
yesan_train_csv=pd.read_csv(path_train_AWS +'예산.csv',encoding='utf-8',index_col=False)
teaan02_train_csv=pd.read_csv(path_train_AWS +'태안.csv',encoding='utf-8',index_col=False)
aan_train_csv=pd.read_csv(path_train_AWS +'홍북.csv',encoding='utf-8',index_col=False)
sung02_train_csv=pd.read_csv(path_train_AWS +'성거.csv',encoding='utf-8',index_col=False)

#=====================================================================================================#
train_aws_li = [sego_train_csv,seyun_train_csv,
    gyeryong_train_csv,Oworld_train_csv,
    jangdong_train_csv,Oworld_train_csv02,
    gongjoo_train_csv,nonsan_train_csv,
    deacheon_train_csv,deasan_train_csv,
    teaan_train_csv,asan_train_csv,
    sung_train_csv,yesan_train_csv,
    teaan02_train_csv,aan_train_csv,sung02_train_csv
]

train_aws_dataset = pd.concat([sego_train_csv,seyun_train_csv,
    gyeryong_train_csv,Oworld_train_csv,
    jangdong_train_csv,Oworld_train_csv02,
    gongjoo_train_csv,nonsan_train_csv,
    deacheon_train_csv,deasan_train_csv,
    teaan_train_csv,asan_train_csv,
    sung_train_csv,yesan_train_csv,
    teaan02_train_csv,aan_train_csv,sung02_train_csv
], axis=0,ignore_index=True)


import os
import pandas as pd

path_train_AWS = "path/to/train/AWS/folder"

train_aws_li = ['세종고운.csv', '세종연서.csv', '계룡.csv', '오월드.csv', '장동.csv', '오월드.csv', 
                '공주.csv', '논산.csv', '대천항.csv', '대산.csv', '태안.csv', '아산.csv', '성거.csv',
                '예산.csv', '태안.csv', '홍북.csv', '성거.csv']

train_aws_dataset = pd.DataFrame()

for file in train_aws_li:
    file_path = os.path.join(path_train_AWS, file)
    df = pd.read_csv(file_path, encoding='utf-8', index_col=False)
    train_aws_dataset = pd.concat([train_aws_dataset, df], axis=0, ignore_index=True)
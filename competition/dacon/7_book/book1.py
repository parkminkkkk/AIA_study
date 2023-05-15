import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout , LSTM
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier,KNeighborsTransformer
from sklearn.metrics import accuracy_score,r2_score
import time
import warnings

warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators

def rmse(y_test,y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

path = "d:/study/_data/dacon_book/"
path_save = "d:/study/_save/dacon_book/"

#1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(871393, 9)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (159621, 8)
'''
Index(['User-ID', 'Book-ID', 'Book-Rating', 'Age', 'Location', 
'Book-Title','Book-Author', 'Year-Of-Publication', 'Publisher'],
      dtype='object')
'''
print(train_csv.info()) 

############################## 'User-ID' 수치형으로 변환 ##############################
astype_list=['User-ID','Book-ID']
for i in astype_list : 
    train_csv[i] = train_csv[i].str[5:].astype(float)
    test_csv[i] = test_csv[i].str[5:].astype(float)
print(train_csv['User-ID'])

####################################### 라벨 인코딩 #####################################

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의
le_list = ['Location', 'Book-Title','Book-Author', 'Publisher']
csv_all = pd.concat([train_csv,test_csv],axis=0)
print(csv_all.columns)
'''
Index(['User-ID', 'Book-ID', 'Book-Rating', 'Age', 'Location', 'Book-Title',
       'Book-Author', 'Year-Of-Publication', 'Publisher'],
      dtype='object')
'''
for i,v in enumerate(le_list) :
    csv_all[v] = le.fit_transform(csv_all[v]) # 0과 1로 변화
    train_csv[v] = le.transform(train_csv[v]) # 0과 1로 변화
    test_csv[v] = le.transform(test_csv[v]) # 0과 1로 변화
    print(f'{i}번째',np.unique(csv_all[v]))
    # #print(np.unique(aaa,return_count=True))
    # test_csv[''] = le.transform(test_csv[''])


#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['Book-Rating'],axis=1)#(1328, 9)
print("x : ", x) # [7500 rows x 7 columns]


y = train_csv['Book-Rating']

print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,random_state=8715
)

print("x_train.shape : ",x_train.shape) #(6750, 9)
print("y_train.shape : ",y_train.shape) #(6750,)
print("x_test.shape : ",x_test.shape) #(750, 9)
print("y_test.shape : ",y_test.shape) # (750,)

# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = MinMaxScaler()
#scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape : ",x_train.shape)
print("y_train.shape : ",y_train.shape)

# x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

####################################### 상관관계 찾기 #####################################
# import matplotlib.pyplot as plt
# import seaborn as sns

# print(test_csv.corr())Epoch 00074: early stopping
# plt.figure(figsize=(10,8))
# sns.set(font_scale=1.2)
# sns.heatmap(train_csv.corr(),square=True, annot=True,cbar=True)
# plt.show()


print('모델 시작')
# 2. 모델 구성
start = time.time()
model = Sequential()

model.add(Dense(128,input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(11,activation="softmax"))


# 3. 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,epochs=12,batch_size=8008)

# 4. 평가, 예측 
# print("최상의 매개변수 : ",model.best_params_)
# print("최상의 매개변수 : ",model.best_score_)
result = model.evaluate(x_test,y_test)
print(" result: ", result)
y_pred= model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
acc = accuracy_score(y_test,y_pred)
print('acc :',acc)
end = time.time()
print('걸린시간:',round(end-start,2))


# 5. 제출

submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
print(submission.shape)
y_submit = model.predict(test_csv)
y_submit=np.argmax(y_submit,axis=1)

print(y_submit.shape)
submission['Book-Rating'] =y_submit
submission.to_csv(path_save+'submit0512_02.csv')
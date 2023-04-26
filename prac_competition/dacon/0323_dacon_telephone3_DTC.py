import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#1. 데이터 

#1-1 데이터 가져오기
path = 'd:/study/_data/dacon_telephone/'
path_save = './_save/dacon_telephone/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) #[30200 rows x 13 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) #[12943 rows x 12 columns] #전화해지여부 제외
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_csv.isnull().sum()) #결측치 없음

#1-2 데이터 정하기
x = train_csv.drop(['전화해지여부'], axis=1) 
y = train_csv['전화해지여부']  #(5497,)

# print(x.shape) (30200, 12)
# print(y.shape) (30200,)

#1-3 onehotencoding
print(np.unique(y))  #[0 1]
y=pd.get_dummies(y)
y = np.array(y)
print(y.shape)     #(30200, 2)


#1-4 데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=640)


#1-5 스케일링
scaler = MaxAbsScaler() 
# scaler = MinMaxScaler() 
# scaler = RobustScaler() 
# scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 

#2. 모델구성 
model = DecisionTreeClassifier(random_state=42)

# 탐색할 파라미터 범위 지정
param_grid = {'max_depth': [1,2,3],
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2,3,4,5],
    'max_features': [None]}

# 그리드 서치 객체 생성
grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring ="average_precision")

# class_weight = {0:2048, 1:209715}

#3. 컴파일, 훈련 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['val_acc'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='f1_score', patience=1000, mode='auto', 
                   verbose=1, 
                   restore_best_weights=True
                   )

# 그리드 서치 수행
grid_search.fit(x_train, y_train)
# model.fit(x_train, y_train)
        #   , epochs=1000, batch_size=32, validation_split=0.1, verbose=1, 
        #   class_weight=class_weight,
        #   callbacks=(es))

#4. 평가, 예측 
# results = model.evaluate(x_test, y_test)
# print('results:', results)  
y_pred = grid_search.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

# 최적의 파라미터 출력
print('Best Parameters:', grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

f1_score = f1_score(y_test, y_pred, average='macro')
print('f1', f1_score)

#5. 파일 생성 

y_submit = grid_search.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)

submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submit_csv['전화해지여부'] = y_submit
print(y_submit)


#시간저장
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submit_csv.to_csv(path_save + 'submit_telephone_DTC' + date + '.csv') 




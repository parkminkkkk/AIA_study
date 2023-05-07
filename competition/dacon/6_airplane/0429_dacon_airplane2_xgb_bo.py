import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import time
# Load data
def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('d:/study/_data/dacon_airplane/train.csv', 'train')
csv_to_parquet('d:/study/_data/dacon_airplane/test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('d:/study/_data/dacon_airplane/sample_submission.csv', index_col = 0)

#print(train)
# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
NaN = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']

for col in NaN:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)

    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])

    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
print('Done.')

print(train.isnull().sum()) #Delay  744999
# Remove unlabeled data
train = train.dropna()
print(train.isnull().sum()) #Delay   0

column4 = {}
for i, column in enumerate(sample_submission.columns):
    column4[column] = i

def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column4))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

pf = PolynomialFeatures(degree=2)
train_x = pf.fit_transform(train_x)

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler2.fit_transform(test_x)


bayesian_params = {
    'learning_rate' : (0.01, 1),
    'max_depth' : (3,16),
    'gamma' : (0,10),        
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),          #subsample 범위 : 0~1사이여야함  min,max / dropout과 비슷한 개념 (훈련을 시킬때의 샘플 양)
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0,1),      #0~1
    'colsample_bynode' : (0, 1),
    'reg_lambda' : (-0.001, 10),      #reg_lambda : 무조건 양수만     max
    'reg_alpha' : (0.01, 50)
}


#모델 정의 
def xgb_hamsu(learning_rate, max_depth, gamma,min_child_weight,subsample,colsample_bytree,colsample_bylevel,colsample_bynode,reg_lambda,reg_alpha):
    params = { 
        'n_estimators' : 1000,
        'learning_rate' : learning_rate,   
        'max_depth' : int(round(max_depth)),
        'gamma' : gamma,
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1), 0),     #무조건 0~1사이 
        'colsample_bytree' : colsample_bytree,  
        'colsample_bylevel' : colsample_bylevel,  
        'colsample_bynode' : colsample_bynode,  
        'reg_lambda' : max(reg_lambda, 0),          #무조건 양수만  (위의 범위에서 -0.01이 선택되어 들어오더라도 여기서 쓸수있는 범위로 변환 '0'으로 바뀌어서 들어감) 
        'reg_alpha' : reg_alpha                                       #-최대한 위에서 파라미터 범위내로 잡아주는게 좋음 
        }
    
    model = XGBClassifier(**params)
    model.fit(train_x, train_y,
              eval_set=[(train_x, train_y), (val_x, val_y)],
              eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50
              )

    results = accuracy_score(val_y, y_predict)
    y_pred = model.predict_proba(test_x)
    submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
    submission.to_csv('./_save/dacon_airplane/xgb_bo.비행기.csv', index=True)
    # submission.to_csv('c:/study/_save/dacon_airplane/1708submission.csv', float_format='%.3f')
    return results

lgbm_bo = BayesianOptimization(f = xgb_hamsu, 
                               pbounds= bayesian_params,
                               random_state=337
                               )



    # y_predict = model.predict(val_x)
    # results = accuracy_score(val_y, y_predict)
    # val_y_pred = model.predict(val_x)
    # accuracy = accuracy_score(val_y, val_y_pred)
    # f1 = f1_score(val_y, val_y_pred, average='weighted')
    # precision = precision_score(val_y, val_y_pred, average='weighted')
    # recall = recall_score(val_y, val_y_pred, average='weighted')
    # logloss = log_loss(val_y, val_y_pred)
    # print(f'Accuracy: {accuracy}')
    # print(f'F1 Score: {f1}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'logloss: {logloss}')


start_time = time.time()
n_iter = 5
lgbm_bo.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(lgbm_bo.max)
print(n_iter, "번 걸린시간:", end_time-start_time)
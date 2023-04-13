import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV

# Load train and test data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
# ...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])
X_test = scaler.transform(test_data.iloc[:, :-1])

# Define the model
model = LocalOutlierFactor()

# Define the hyperparameters to search over
param_grid = {'n_neighbors': [5, 10, 15, 20, 25],
              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
              'contamination': [0.01, 0.02, 0.03, 0.04, 0.05]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc')
grid_search.fit(X_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Model Definition with best hyperparameters
best_params = grid_search.best_params_
model = LocalOutlierFactor(n_neighbors=best_params['n_neighbors'], algorithm=best_params['algorithm'], contamination=best_params['contamination'])

# Model Training
model.fit(X_train)

# Model Prediction
y_pred = model.fit_predict(X_test)
submission['label'] = [1 if label == -1 else 0 for label in y_pred]
print(submission.value_counts())
print(submission['label'].value_counts())

# Save the results to a submission file
#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  
submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)

'''
#gridsearch/ 'ball_tree',5, 0.01
0    7315
1      74
Name: label, dtype: int64
=>
'''

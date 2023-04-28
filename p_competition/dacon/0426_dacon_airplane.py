#준지도학습
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import log_loss

#1. 데이터
path = 'd:/study/_data/dacon_airplane/'
path_save = './_save/dacon_airplane/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)  #[1000000 rows x 18 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)  #[1000000 rows x 17 columns]

# print(train_csv.describe())
# print(train_csv.columns)
'''
Index(['Month', 'Day_of_Month', 'Estimated_Departure_Time',
       'Estimated_Arrival_Time', 'Cancelled', 'Diverted', 'Origin_Airport',
       'Origin_Airport_ID', 'Origin_State', 'Destination_Airport',
       'Destination_Airport_ID', 'Destination_State', 'Distance', 'Airline',
       'Carrier_Code(IATA)', 'Carrier_ID(DOT)', 'Tail_Number', 'Delay'],
      dtype='object')
'''

x = train_csv.drop(['Delay','Cancelled', 'Diverted'], axis=1)
y = train_csv['Delay']
test_csv = test_csv.drop(['Cancelled', 'Diverted'], axis=1)


# # Generate a random binary classification dataset
# x, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
#                            n_redundant=0, n_clusters_per_class=2, random_state=42)

# Split the dataset into labeled and unlabeled sets
labeled_indices = [0, 1, 2, 3, 4, 5]
unlabeled_indices = [i for i in range(len(x)) if i not in labeled_indices]

#2. 모델
model = XGBClassifier()

# Create a self-training classifier and fit it on the labeled data
self_training_model = SelfTrainingClassifier(model, threshold=0.9, max_iter=100)
self_training_model.fit(x[labeled_indices], y[labeled_indices])

# Predict the labels of the unlabeled data
predicted_labels = self_training_model.predict(x[unlabeled_indices])

# Evaluate the performance of the self-training model on the entire dataset
score = self_training_model.score(x, y)
print(f"Accuracy: {score:.3f}")

# Use the trained model to predict the labels and probabilities of the test data
predicted_labels = self_training_model.predict(test_csv)
predicted_probs = self_training_model.predict_proba(test_csv)
# Evaluate the performance of the self-training model using log loss
score = log_loss(y, predicted_probs)

# Create a DataFrame with the predicted labels
y_submit = pd.DataFrame(predicted_labels, columns=['Delayed'])

# Add the probabilities of the predicted labels to the DataFrame
y_submit['Not_Delayed'] = predicted_probs[:,0]

# Load the sample submission file and set the 'Delayed' and 'Not_Delayed' columns
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Delayed'] = y_submit['Delayed']
submission['Not_Delayed'] = y_submit['Not_Delayed']

# Save the submission file
submission.to_csv('submission.csv')





print(f"Log loss: {score:.3f}")
#######################################################################
y_submit = model.predict(test_csv)
y_submit = pd.DataFrame(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Delayed'] = y_submit[:,1]
submission['Not_Delayed'] = y_submit[:,0]

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(path_save + 'dacon_airplane' + date + '.csv')


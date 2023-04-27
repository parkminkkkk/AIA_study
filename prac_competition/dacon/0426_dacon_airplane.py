#준지도학습
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
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


# Generate a random binary classification dataset
x, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=0, n_clusters_per_class=2, random_state=42)

# Split the dataset into labeled and unlabeled sets
labeled_indices = [0, 1, 2, 3, 4, 5]
unlabeled_indices = [i for i in range(len(x)) if i not in labeled_indices]

# Create a logistic regression model
model = LogisticRegression()

# Create a self-training classifier and fit it on the labeled data
self_training_model = SelfTrainingClassifier(model, threshold=0.9, max_iter=100)
self_training_model.fit(x[labeled_indices], y[labeled_indices])

# Predict the labels of the unlabeled data
predicted_labels = self_training_model.predict(x[unlabeled_indices])

# Evaluate the performance of the self-training model on the entire dataset
score = self_training_model.score(x, y)
print(f"Accuracy: {score:.3f}")

# Predict the labels and probabilities of the unlabeled data
predicted_labels = self_training_model.predict(x[unlabeled_indices])
predicted_probs = self_training_model.predict_proba(x[unlabeled_indices])

# Combine the labeled and unlabeled data
combined_x = np.concatenate([x[labeled_indices], x[unlabeled_indices]])
combined_y = np.concatenate([y[labeled_indices], predicted_labels])
combined_probs = np.concatenate([np.zeros((len(labeled_indices), 2)), predicted_probs])

# Evaluate the performance of the self-training model using log loss
score = log_loss(combined_y, combined_probs)

print(f"Log loss: {score:.3f}")


# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

y_submit = model.predict(test_csv)
import numpy as np
import pandas as pd
y_submit = pd.DataFrame(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Delayed'] = y_submit
submission['Not_Delayed'] = y_submit

submission.to_csv(path_save + 'kaggle_house_' + date + '.csv')


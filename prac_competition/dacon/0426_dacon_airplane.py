#준지도학습

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

#1. 데이터
path = 'd:/study/_data/dacon_airplane/'
path_save = './_save/dacon_airplane/'


# Generate a random binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=0, n_clusters_per_class=2, random_state=42)

# Split the dataset into labeled and unlabeled sets
labeled_indices = [0, 1, 2, 3, 4, 5]
unlabeled_indices = [i for i in range(len(X)) if i not in labeled_indices]

# Create a logistic regression model
model = LogisticRegression()

# Create a self-training classifier and fit it on the labeled data
self_training_model = SelfTrainingClassifier(model, strategy='most_confident')
self_training_model.fit(X[labeled_indices], y[labeled_indices])

# Predict the labels of the unlabeled data
predicted_labels = self_training_model.predict(X[unlabeled_indices])

# Evaluate the performance of the self-training model on the entire dataset
score = self_training_model.score(X, y)

print(f"Accuracy: {score:.3f}")
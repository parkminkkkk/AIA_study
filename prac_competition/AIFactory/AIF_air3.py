import pandas as pd
from sklearn.cluster import KMeans

# Load training data
path='./_data/AIFac_air/'
save_path= './_save/AIFac_air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Preprocess data
# ...

# Train k-means clustering model
kmeans = KMeans(n_clusters=300, random_state=42)
kmeans.fit(train_data.drop('type', axis=1))

# Make predictions on test set and save results
X_test = test_data.drop('type', axis=1)
y_pred = kmeans.predict(X_test)

submission = pd.DataFrame({'type': test_data['type'], 'label': y_pred})
submission.to_csv(save_path+'submission2.csv', index=False)



# Import necessary libraries
from sklearn.ensemble import IsolationForest
import numpy as np

# Smaple data (normal data points clustered around 0)
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# New test data including some outliers
X_test = np.r_[X  + 2, X - 2, np.random.uniform(low=-6, high=6, size=(20, 2))]

# Initialise and train the model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict on test data (-1 indicates an anomaly, 1 indicates normal)
predictions = model.predict(X_test)

# Display predictions
print("Predictions:", predictions)
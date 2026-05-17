from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load Iris dataset
data = load_iris()
X  = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset information
print(X.head)

print("Dataset Info:")
print(X.describe())
print("\n Target Classes:", data.target_names)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predit and evaluate
y_pred = knn.predict(X_test)
print("Accuracy without Scaling:", accuracy_score(y_test, y_pred))

# Apply Min-max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train k-NN classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy with MinMax Scaling:", accuracy_score(y_test_scaled, y_pred_scaled))
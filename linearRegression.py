from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


# sample data (e.g. house size vs house price)
X = np.array([[1000], [1200], [1400], [1600], [1800], [2000], [2200], [2400], [2600], [2800]])
y = np.array([210000, 235000, 290000, 315000, 370000, 395000, 450000, 475000, 530000, 555000])

# Split the data into training anf testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)